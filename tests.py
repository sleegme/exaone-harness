"""Offline contract tests. Run with python3 tests.py; no test dependencies."""

from __future__ import annotations

import json
import io
import http.client
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator
from unittest.mock import patch

from agent import (
    Budgets, Document, ExaClient, Executor, JSON, ReplayClient, RequestFailed, RetryPolicy,
    Sandbox, TransportError, mapping, run_agent,
)

ROOT = Path(__file__).parent
GUIDE = ROOT.joinpath("guide.txt").read_text()


def reply(
    content: str | None = "done", finish: str = "stop",
    calls: list[JSON] | None = None, role: str = "assistant",
) -> Document:
    return {
        "choices": [{
            "finish_reason": finish,
            "message": {"role": role, "content": content, "tool_calls": calls or []},
        }],
        "usage": {
            "prompt_tokens": 13, "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


def call(name: str, arguments: Document, identifier: str = "c1") -> Document:
    return {
        "id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def read(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def termination() -> None:
    """Regressions: truncation/empty/tool/turn-limit accidentally published as final."""
    cases = [
        (reply("partial", "length"), "length"),
        (reply("  "), "empty_reply"),
        (reply(None), "empty_reply"),
        (reply("hidden</think>answer"), "reasoning_in_final"),
        (reply("denied", "content_filter"), "content_filter"),
        (reply("tool output", role="tool"), "protocol: expected assistant role, got tool"),
        (reply(None, "tool_calls", [call("run", {"cmd": "printf verified"})]), "turn_limit"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for response, failure in cases:
            result = run_agent(
                "task", Sandbox(Path(directory)), Executor(ReplayClient([response])),
                GUIDE, max_turns=1,
            )
            assert result["status"] == "failed", failure
            assert result["failure"] == failure
            assert result["final"] is None
            record = mapping(result["requests"][0])
            assert record["failure"] == failure
            assert record["usage"] == response["usage"]
            assert record["role"] == mapping(mapping(response["choices"][0])["message"])["role"]
            assert record["finish_reason"] == mapping(response["choices"][0])["finish_reason"]
            assert record["has_tools"] == bool(mapping(mapping(response["choices"][0])["message"])["tool_calls"])
            if failure == "turn_limit":
                assert result["transcript"][-1]["role"] == "tool"
        # A length-finished tool call must never have side effects.
        result = run_agent("task", Sandbox(Path(directory)), Executor(ReplayClient([
            reply(None, "length", [call("write_file", {"path": "bad", "content": "bad"})]),
        ])), GUIDE)
        assert not Path(directory, "bad").exists()
        assert result["tools"] == []


def independent_budgets() -> None:
    """The independent client can spend its reasoning cap without eating final space."""
    class BudgetClient:
        separate_budgets = True

        def complete(self, payload: Document, timeout: float) -> Document:
            del timeout
            reasoning_demand, final_demand = 1000, 12
            # Simulate independent provider allocation; a shared max_tokens fails.
            reasoning_spent = min(reasoning_demand, payload["reasoning_max_tokens"])
            final_spent = min(final_demand, payload["max_tokens"])
            return reply(
                f"reasoning={reasoning_spent};final={final_spent}",
                "stop" if final_spent == final_demand else "length",
            )

    with tempfile.TemporaryDirectory() as directory:
        result = run_agent("task", Sandbox(Path(directory)), Executor(BudgetClient()),
                           GUIDE, budgets=Budgets(5, 12))
        assert result["final"] == "reasoning=5;final=12"
        assert result["budget"]["mode"] == "independent"
    # Unsupported providers must advertise a shared total, never a fictitious cap.
    shared = Budgets(5, 12).parameters(False)
    assert shared["max_tokens"] == 17
    assert "reasoning_max_tokens" not in shared
    exhausted = Budgets(0, 12).parameters(False)
    assert exhausted["max_tokens"] == 12
    assert exhausted["chat_template_kwargs"] == {"enable_thinking": False}


def delta(content: Document, finish: str | None = None) -> Document:
    return {"choices": [{"index": 0, "delta": content, "finish_reason": finish}]}


class SocketReplay:
    """The small socket surface exposed by urllib's HTTPResponse."""

    def __init__(self) -> None:
        self.timeout = 2.0
        self.timeouts: list[float] = []
        self.shutdowns = 0
        self.closed = False

    def settimeout(self, value: float) -> None:
        self.timeout = value
        self.timeouts.append(value)

    def shutdown(self, how: int) -> None:
        del how
        self.shutdowns += 1

    def close(self) -> None:
        self.closed = True


class HTTPReplay(io.BytesIO):
    """Replay urllib's iterable body and underlying socket without network I/O."""

    def __init__(self, wire: bytes):
        super().__init__(wire)
        self.socket = SocketReplay()
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.socket))
        self.stopped_at = 0
        self.consumed = 0

    def __next__(self) -> bytes:
        raw = super().__next__()
        self.consumed += 1
        return raw

    def close(self) -> None:
        if not self.closed:
            self.stopped_at = self.tell()
        super().close()


class StreamingReplayClient:
    """Lazy stream replay with closure/consumption evidence and no network."""

    separate_budgets = False

    def __init__(self, scripts: list[list[Document]]):
        self.scripts = iter(scripts)
        self.payloads: list[Document] = []
        self.consumed: list[int] = []
        self.closed = 0

    @contextmanager
    def stream(
        self, payload: Document, timeout: float, *,
        deadline: float, clock: Callable[[], float],
    ) -> Iterator[Iterator[Document]]:
        assert timeout > 0
        assert deadline > clock()
        self.payloads.append(payload)
        self.consumed.append(0)
        try:
            script = next(self.scripts)
        except StopIteration:
            raise AssertionError("unexpected model request after replay exhausted") from None

        def events() -> Iterator[Document]:
            for event in script:
                self.consumed[-1] += 1
                if "error" in event:
                    error = mapping(event["error"])
                    raise TransportError(error["status"], str(error["retry_after"]))
                yield event

        try:
            yield events()
        finally:
            self.closed += 1


def streaming_reasoning_budget() -> None:
    """Crossing reasoning cap closes early, never executes pending calls or retries."""
    pending = call("write_file", {"path": "bad", "content": "must not run"})
    client = StreamingReplayClient([[
        delta({"reasoning_content": "x"}),
        delta({"tool_calls": [{**pending, "index": 0}]}),
        delta({"reasoning_content": "yz"}),
        delta({"content": "this must not be consumed"}, "stop"),
    ]])
    with tempfile.TemporaryDirectory() as directory:
        result = run_agent("task", Sandbox(Path(directory)), Executor(client),
                           GUIDE, budgets=Budgets(2, 64))
        assert not Path(directory, "bad").exists()
    assert result["failure"] == "reasoning_budget_exceeded"
    assert result["final"] is None
    assert result["requests"][0]["finish_reason"] == "reasoning_budget_exceeded"
    assert result["requests"][0]["budget_usage"]["reasoning"] == 3
    assert len(result["transport_attempts"]) == 1
    assert result["tools"] == []
    assert client.consumed == [3]
    assert client.closed == 1


def streaming_answer_budget() -> None:
    """Reasoning allowance cannot subsidize an oversized streamed answer."""
    client = StreamingReplayClient([[
        delta({"reasoning_content": "abcd"}),
        delta({"content": "가"}),
        delta({"content": "a"}),
        delta({"content": "ignored"}, "stop"),
    ]])
    with tempfile.TemporaryDirectory() as directory:
        result = run_agent("task", Sandbox(Path(directory)), Executor(client),
                           GUIDE, budgets=Budgets(4, 3))
    assert result["failure"] == "final_budget_exceeded"
    assert result["final"] is None
    assert result["requests"][0]["finish_reason"] == "final_budget_exceeded"
    assert result["requests"][0]["budget_usage"]["answer"] == 4
    assert result["requests"][0]["budget_usage"]["reasoning"] == 4
    assert client.consumed == [3]
    assert client.closed == 1


def streaming_cumulative_budget() -> None:
    """Exact exhaustion across tool turns disables thinking on the next request."""
    first = call("write_file", {"path": "x", "content": "first"})
    client = StreamingReplayClient([
        [
            delta({"reasoning_content": "a"}),
            delta({"tool_calls": [{**first, "index": 0}]}, "tool_calls"),
        ],
        [
            delta({"reasoning_content": "bc"}),
            delta({"tool_calls": [{
                "index": 0, "id": "c2", "type": "function",
                "function": {"name": "write_file", "arguments": '{"path":'},
            }]}),
            delta({"tool_calls": [{
                "index": 0, "function": {"arguments": '"x", "content":"second"}'},
            }]}, "tool_calls"),
        ],
        [delta({"content": "done"}, "stop")],
    ])
    with tempfile.TemporaryDirectory() as directory:
        result = run_agent("task", Sandbox(Path(directory)), Executor(client),
                           GUIDE, budgets=Budgets(3, 128))
        assert Path(directory, "x").read_text() == "second"
    assert result["status"] == "succeeded"
    assert result["final"] == "done"
    assert result["budget"]["mode"] == "stream_enforced"
    assert result["budget"]["reasoning_used"] == 3
    assert [p["chat_template_kwargs"]["enable_thinking"] for p in client.payloads] == [True, True, False]
    assert [p["max_tokens"] for p in client.payloads] == [131, 130, 128]
    assert [r["budget_usage"]["reasoning_total"] for r in result["requests"]] == [1, 3, 3]
    assert client.closed == 3


def streaming_retry_keeps_reasoning_charge() -> None:
    """Partial streams cannot replenish the run's reasoning budget on retry."""
    first = call("list_dir", {"path": "."})
    client = StreamingReplayClient([
        [delta({"reasoning_content": "ab"}), {"error": {"status": 429, "retry_after": "0"}}],
        [
            delta({"reasoning_content": "c"}),
            delta({"tool_calls": [{**first, "index": 0}]}, "tool_calls"),
        ],
        [delta({"content": "done"}, "stop")],
    ])
    clock = Clock()
    with tempfile.TemporaryDirectory() as directory:
        result = run_agent("task", Sandbox(Path(directory)), Executor(
            client, clock=clock.read, sleep=clock.advance,
        ), GUIDE, budgets=Budgets(3, 128))
    assert result["status"] == "succeeded"
    assert result["budget"]["reasoning_used"] == 3
    assert [a["status"] for a in result["transport_attempts"]] == [429, 200, 200]
    assert [a["reasoning_budget_used"] for a in result["transport_attempts"]] == [2, 1, 0]
    assert [p["max_tokens"] for p in client.payloads] == [131, 129, 128]
    assert client.payloads[-1]["chat_template_kwargs"]["enable_thinking"] is False
    assert len(result["requests"]) == 2
    assert client.closed == 3


def exa_streaming_wire_budget() -> None:
    """The real ExaClient uses SSE and closes HTTP before reading over-budget tail."""
    events = [
        delta({"role": "assistant", "reasoning_content": "x"}),
        delta({"reasoning_content": "yz"}),
        delta({"content": "never read"}, "stop"),
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    wire += b"data: [DONE]\n\n"

    http = HTTPReplay(wire)
    with tempfile.TemporaryDirectory() as directory:
        with patch("agent.urllib.request.OpenerDirector.open", return_value=http) as open_http:
            result = run_agent("task", Sandbox(Path(directory)), Executor(ExaClient("offline")),
                               GUIDE, budgets=Budgets(2, 64))
        body = json.loads(open_http.call_args.args[0].data)
    assert body["stream"] is True
    assert body["stream_options"]["include_usage"] is True
    assert "reasoning_max_tokens" not in body
    assert result["failure"] == "reasoning_budget_exceeded"
    assert result["final"] is None
    assert http.closed
    assert http.stopped_at < len(wire)


def exa_streaming_wire_completion() -> None:
    """Real SSE parsing preserves usage arriving after the terminal delta."""
    events = [
        delta({"role": "assistant", "reasoning_content": "ab"}),
        delta({"content": "do"}),
        delta({"content": "ne"}, "stop"),
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 6}},
    ]
    wire = b": heartbeat\n\n" + b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    ) + b"data: [DONE]\n\n"
    http = HTTPReplay(wire)
    with tempfile.TemporaryDirectory() as directory:
        with patch("agent.urllib.request.OpenerDirector.open", return_value=http):
            result = run_agent("task", Sandbox(Path(directory)), Executor(ExaClient("offline")),
                               GUIDE, budgets=Budgets(2, 4))
    assert result["status"] == "succeeded"
    assert result["final"] == "done"
    assert result["requests"][0]["usage"] == {"prompt_tokens": 9, "completion_tokens": 6}
    assert result["requests"][0]["budget_usage"]["answer"] == 4
    assert result["budget"]["reasoning_used"] == 2
    assert http.closed


def raw_stream_deadline() -> None:
    """Comments, empty lines and unfinished events cannot outlive the deadline."""
    for line in (b": heartbeat\n", b"\n", b"data: {}\n"):
        clock = Clock()

        class SlowHTTP(HTTPReplay):
            def __next__(self) -> bytes:
                raw = super().__next__()
                clock.advance(0.5)
                return raw

        terminal = b"data: " + json.dumps(delta({"content": "done"}, "stop")).encode() + b"\n\n"
        http = SlowHTTP(line * 10 + terminal)
        with tempfile.TemporaryDirectory() as directory:
            with patch("agent.urllib.request.OpenerDirector.open", return_value=http):
                result = run_agent("task", Sandbox(Path(directory)), Executor(
                    ExaClient("offline"), RetryPolicy(2, 2),
                    clock=clock.read, sleep=clock.advance,
                ), GUIDE)
        assert clock.now == 2, (line, clock.now)
        assert http.consumed == 4
        assert http.closed
        assert result["failure"] == "retry_time_budget"
        assert result["final"] is None
        assert len(result["transport_attempts"]) == 1
        assert result["transport_attempts"][0]["error"] == "transport_timeout"
        assert result["model_seconds"] == 2


def silent_stream_remaining_timeout() -> None:
    """A read after heartbeats gets only the remaining time, not a fresh timeout."""
    clock = Clock()

    class SilentHTTP(HTTPReplay):
        def __next__(self) -> bytes:
            if self.consumed < 2:
                raw = super().__next__()
                clock.advance(0.5 if self.consumed == 1 else 0.25)
                return raw
            # Simulate a socket blocking exactly until its configured read timeout.
            clock.advance(self.socket.timeout)
            raise TimeoutError("silent replay socket")

    http = SilentHTTP(b": heartbeat\n\n")
    with tempfile.TemporaryDirectory() as directory:
        with patch("agent.urllib.request.OpenerDirector.open", return_value=http):
            result = run_agent("task", Sandbox(Path(directory)), Executor(
                ExaClient("offline"), RetryPolicy(2, 2),
                clock=clock.read, sleep=clock.advance,
            ), GUIDE)
    assert clock.now == 2
    assert http.socket.timeouts == [2, 2, 1.5, 1.25]
    assert http.closed
    assert result["failure"] == "retry_time_budget"
    assert result["final"] is None
    assert result["transport_attempts"][0]["error"] == "transport_timeout"
    assert result["model_seconds"] == 2


def stream_timeouts_share_retry_budget() -> None:
    """An attempt timeout and its backoff reduce the same total request budget."""
    clock = Clock()
    responses: list[HTTPReplay] = []

    class SlowHTTP(HTTPReplay):
        def __next__(self) -> bytes:
            raw = super().__next__()
            clock.advance(0.5)
            return raw

    def open_http(request, timeout: float) -> HTTPReplay:
        del request
        assert timeout == 2
        http = SlowHTTP(b": heartbeat\n" * 20)
        responses.append(http)
        return http

    with tempfile.TemporaryDirectory() as directory:
        with patch("agent.urllib.request.OpenerDirector.open", side_effect=open_http):
            result = run_agent("task", Sandbox(Path(directory)), Executor(
                ExaClient("offline"), RetryPolicy(5, 2),
                clock=clock.read, sleep=clock.advance,
            ), GUIDE)
    assert clock.now == 5
    assert result["failure"] == "retry_time_budget"
    assert result["final"] is None
    assert [a["error"] for a in result["transport_attempts"]] == [
        "transport_timeout", "transport_timeout",
    ]
    assert result["model_seconds"] == 4
    assert result["retry_wait_seconds"] == 1
    assert [http.consumed for http in responses] == [4, 4]
    assert all(http.closed for http in responses)


def watchdog_interrupts_partial_line() -> None:
    """The absolute watchdog shuts down a read kept alive without a newline."""
    clock = Clock()
    timers = []

    class TimerReplay:
        def __init__(self, interval: float, action: Callable[[], None]):
            self.interval, self.action = interval, action
            self.daemon = False
            self.started = self.cancelled = self.joined = False
            timers.append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

        def join(self) -> None:
            self.joined = True

    class PartialHTTP(HTTPReplay):
        def __next__(self) -> bytes:
            assert len(timers) == 1 and timers[0].started
            # No complete raw line is returned before the absolute alarm fires.
            clock.advance(timers[0].interval - clock.now)
            timers[0].action()
            assert self.socket.shutdowns == 1
            raise OSError("read interrupted by socket shutdown")

    http = PartialHTTP(b"")

    def open_http(request, timeout: float) -> HTTPReplay:
        del request
        assert timeout == 2
        clock.advance(0.5)  # Response headers also consume the attempt's deadline.
        return http

    with tempfile.TemporaryDirectory() as directory:
        with patch("agent.urllib.request.OpenerDirector.open", side_effect=open_http):
            with patch("agent.threading.Timer", TimerReplay):
                result = run_agent("task", Sandbox(Path(directory)), Executor(
                    ExaClient("offline"), RetryPolicy(2, 10),
                    clock=clock.read, sleep=clock.advance,
                ), GUIDE)
    assert clock.now == 2
    assert timers[0].interval == 2
    assert timers[0].cancelled and timers[0].joined
    assert http.socket.timeouts == [1.5, 1.5]
    assert http.closed
    assert result["failure"] == "retry_time_budget"
    assert result["transport_attempts"][0]["error"] == "transport_timeout"
    assert result["final"] is None


def localhost_request_deadline() -> None:
    """Real urllib must interrupt slow headers as well as every body read phase."""
    for mode in ("slow_headers", "silent_body", "partial_body", "heartbeat_body", "normal"):
        stop, phase_started = threading.Event(), threading.Event()
        errors: list[str] = []
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(3)
            address = server.getsockname()

            def serve() -> None:
                try:
                    peer, _ = server.accept()
                    with peer:
                        peer.settimeout(3)
                        request = b""
                        while b"\r\n\r\n" not in request:
                            block = peer.recv(8192)
                            if not block:
                                return
                            request += block
                        if mode == "slow_headers":
                            peer.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                            phase_started.set()
                            for _ in range(50):
                                if stop.wait(0.02):
                                    return
                                peer.sendall(b"x")
                            peer.sendall(b"\r\nContent-Length: 0\r\n\r\n")
                        else:
                            peer.sendall(
                                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
                            )
                            phase_started.set()
                            if mode == "normal":
                                event = delta({"content": "done"}, "stop")
                                peer.sendall(
                                    b"data: " + json.dumps(event).encode() + b"\n\n"
                                    b"data: [DONE]\n\n"
                                )
                            elif mode == "silent_body":
                                stop.wait(2)
                            else:
                                while not stop.wait(0.02):
                                    peer.sendall(
                                        b": beat\n\n" if mode == "heartbeat_body" else b":x"
                                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Expected when the deadline closes the client socket.
                except (OSError, ValueError) as error:
                    errors.append(str(error))

            def resolve(host, port, *args, **kwargs):
                del args, kwargs
                assert host == "api.exaone.ai" and port == 443
                return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)]

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            # Use real HTTPConnection, urllib and socket reads on loopback. TLS
            # alone is bypassed; no DNS query or external connection is possible.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            budget = 2 if mode == "normal" else 0.15
            try:
                with tempfile.TemporaryDirectory() as directory:
                    with patch("agent.socket.getaddrinfo", side_effect=resolve):
                        with patch.object(
                            http.client.HTTPSConnection, "connect", http.client.HTTPConnection.connect,
                        ):
                            with patch("agent.urllib.request._opener", opener):
                                started = time.monotonic()
                                result = run_agent("task", Sandbox(Path(directory)), Executor(
                                    ExaClient("offline"), RetryPolicy(budget, budget, max_attempts=1),
                                ), GUIDE)
                                elapsed = time.monotonic() - started
            finally:
                stop.set()
                worker.join(timeout=3)
                assert not worker.is_alive()
            assert phase_started.is_set(), (mode, errors)
            assert not errors, (mode, errors)
            if mode == "normal":
                assert result["final"] == "done"
                assert result["status"] == "succeeded"
            else:
                assert elapsed < 0.45, (mode, elapsed)
                assert result["failure"] == "retry_time_budget", (mode, result["failure"])
                assert result["transport_attempts"][0]["error"] == "transport_timeout"
                assert result["final"] is None
                assert len(result["transport_attempts"]) == 1


def transport_retries() -> None:
    """429s replay the same logical request and remain outside model transcript."""
    clock = Clock()
    class TimedClient(ReplayClient):
        def complete(self, payload: Document, timeout: float) -> Document:
            assert timeout <= 20
            clock.advance(2)
            return super().complete(payload, timeout)

    client = TimedClient([
        {"error": {"status": 429, "retry_after": "3"}},
        {"error": {"status": 503, "retry_after": "Thu, 01 Jan 1970 00:00:09 GMT"}},
        reply(),
    ])
    executor = Executor(client, RetryPolicy(20, 20), clock=clock.read,
                        sleep=clock.advance, wall_clock=clock.read)
    with tempfile.TemporaryDirectory() as directory:
        result = run_agent("task", Sandbox(Path(directory)), executor, GUIDE)
    assert result["status"] == "succeeded"
    assert len(result["requests"]) == 1
    assert [m["role"] for m in result["transcript"]] == ["system", "user", "assistant"]
    attempts = result["transport_attempts"]
    assert [a["status"] for a in attempts] == [429, 503, 200]
    assert [a["wait_seconds"] for a in attempts] == [3, 2, 0]
    assert [a["retransmission"] for a in attempts] == [False, True, True]
    assert len({a["attempt_id"] for a in attempts}) == 3
    assert len({a["request_id"] for a in attempts}) == 1
    assert all(a["started_at"] <= a["ended_at"] for a in attempts)
    assert result["model_seconds"] == 6
    assert result["retry_wait_seconds"] == 5
    assert result["wall_seconds"] == 11


def retry_budget_and_status() -> None:
    """Executor refuses Retry-After past deadline and nonretryable status codes."""
    for status, after, failure in [
        (429, "999", "retry_time_budget"),
        (401, "0", "transport_exhausted"),
    ]:
        clock = Clock()
        executor = Executor(
            ReplayClient([{"error": {"status": status, "retry_after": after}}]),
            RetryPolicy(2, 2), clock=clock.read, sleep=clock.advance,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_agent("task", Sandbox(Path(directory)), executor, GUIDE)
        assert result["failure"] == failure
        assert result["final"] is None
        assert len(executor.attempts) == 1
        assert clock.now == 0
        assert result["requests"][0]["usage"] is None
    clock = Clock()
    executor = Executor(ReplayClient([{"error": {"status": 429, "retry_after": "0"}}] * 2),
                        RetryPolicy(20, 20, max_attempts=2), clock=clock.read,
                        sleep=clock.advance)
    try:
        executor.request({}, "limited")
    except RequestFailed as error:
        assert error.reason == "transport_exhausted"
    else:
        raise AssertionError("attempt limit not enforced")
    assert len(executor.attempts) == 2


def concurrency_limit() -> None:
    """Real threads sharing the executor cannot enter the client concurrently."""
    entered, release = threading.Event(), threading.Event()
    second_started = threading.Event()
    class BlockingClient:
        separate_budgets = True

        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0
            self.count = 0
            self.lock = threading.Lock()

        def complete(self, payload: Document, timeout: float) -> Document:
            del payload
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.count += 1
            entered.set()
            assert release.wait(timeout)
            with self.lock:
                self.active -= 1
            return reply()

    client = BlockingClient()
    executor = Executor(client, RetryPolicy(5, 5, concurrency=1))
    def second() -> Document:
        second_started.set()
        return executor.request({}, "two")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.request, {}, "one")
        assert entered.wait(2)
        other = pool.submit(second)
        assert second_started.wait(2)
        # Attempt to acquire the actual shared gate while the first owns it.
        acquired = executor.limit.acquire(blocking=False)
        if acquired:
            executor.limit.release()
        assert not acquired
        release.set()
        first.result(timeout=3)
        other.result(timeout=3)
    assert client.maximum == 1
    assert client.count == 2


def separate_times() -> None:
    """Tool execution and provider attempts contribute to separate counters."""
    clock = Clock()
    class TimedClient(ReplayClient):
        def complete(self, payload: Document, timeout: float) -> Document:
            clock.advance(7)
            return super().complete(payload, timeout)

    class TimedSandbox(Sandbox):
        def execute(self, name: str, arguments: Document) -> str:
            clock.advance(3)
            return super().execute(name, arguments)

    with tempfile.TemporaryDirectory() as directory:
        client = TimedClient([
            reply(None, "tool_calls", [call("write_file", {"path": "x", "content": "ok"})]),
            reply(),
        ])
        result = run_agent("task", TimedSandbox(Path(directory)),
                           Executor(client, clock=clock.read, sleep=clock.advance), GUIDE)
        assert Path(directory, "x").read_text() == "ok"
    assert result["model_seconds"] == 14
    assert result["tool_seconds"] == 3
    assert result["wall_seconds"] == 17
    assert result["tools"][0]["tool_seconds"] == 3
    assert [r["usage"]["completion_tokens"] for r in result["requests"]] == [7, 7]


def sandbox_boundary() -> None:
    """Actual tools deny traversal, symlink escape, host files and inherited secrets."""
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        work = parent / "task"
        work.mkdir()
        secret = parent / "secret"
        secret.write_text("outside sentinel")
        (work / "escape").symlink_to(secret)
        sandbox = Sandbox(work)
        for path in ("../secret", str(secret), "escape"):
            assert sandbox.execute("read_file", {"path": path}).startswith("ERROR:")
            assert sandbox.execute("write_file", {"path": path, "content": "changed"}).startswith("ERROR:")
        assert secret.read_text() == "outside sentinel"
        sandbox.execute("write_file", {"path": "/work/inside.txt", "content": "inside"})
        assert "inside" in sandbox.execute("read_file", {"path": "inside.txt"})
        command = (
            f"test ! -e {secret}; a=$?; "
            "test ! -e /tmp/exa-bench; b=$?; "
            "test ! -e /root; c=$?; "
            "test -z \"$EXAONE_API_KEY\"; d=$?; "
            "test ! -r escape; e=$?; "
            "printf '%s %s %s %s %s\\n' \"$a\" \"$b\" \"$c\" \"$d\" \"$e\"; "
            "cat inside.txt"
        )
        result = sandbox.execute("run", {"cmd": command})
        assert result == "exit 0\n0 0 0 0 0\ninside", result
        result = sandbox.execute("run", {"cmd": "printf changed > /usr/escape-test"})
        assert result.startswith("exit 1\n") and "Read-only file system" in result, result
        for method, args in [
            ("list_dir", {"path": ".."}), ("grep", {"path": "..", "pattern": "."}),
        ]:
            assert sandbox.execute(method, args).startswith("ERROR:")


def duplicate_guard() -> None:
    """Identical checks are blocked until file contents, paths or mode change."""
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        (work / "source.txt").write_text("v1")
        sandbox = Sandbox(work)
        response = reply(None, "tool_calls", [call("run", {"cmd": "cat source.txt"})])
        result = run_agent("task", sandbox, Executor(ReplayClient([
            response, response,
            reply(None, "tool_calls", [call("write_file", {"path": "source.txt", "content": "v2"})]),
            response, reply(),
        ])), GUIDE)
        outputs = [t["result"] for t in result["tools"] if t["name"] == "run"]
        assert outputs == ["exit 0\nv1",
                           "UNCHANGED: identical check blocked; previous result: exit 0\nv1",
                           "exit 0\nv2"]
        # A distinct command has its own result, and failed checks are guarded too.
        assert sandbox.shell("cat missing").startswith("exit 1\n")
        assert sandbox.shell("cat missing").startswith("UNCHANGED:")
        (work / "missing").write_text("created")
        assert sandbox.shell("cat missing") == "exit 0\ncreated"
        assert result["status"] == "succeeded"


def cli_end_to_end() -> None:
    """One CLI task reads, edits, executes a real test, blocks its replay, and finishes."""
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        work = parent / "work"
        work.mkdir()
        (work / "add.py").write_text("def add(a, b):\n    return a - b\n")
        (work / "check.py").write_text(
            "from add import add\nassert add(2, 3) == 5\nprint('tiny test passed')\n"
        )
        task = parent / "task.txt"
        task.write_text("Fix add.py and run python3 check.py. Finish with a one-line answer.")
        events = [
            {"error": {"status": 429, "retry_after": "0"}},
            reply(None, "tool_calls", [call("read_file", {"path": "add.py"})]),
            reply(None, "tool_calls", [call("write_file", {
                "path": "add.py", "content": "def add(a, b):\n    return a + b\n",
            })]),
            reply(None, "tool_calls", [call("run", {"cmd": "python3 check.py"})]),
            reply(None, "tool_calls", [call("run", {"cmd": "python3 check.py"})]),
            reply("add.py fixed; tiny test passed."),
        ]
        replay = parent / "replay.json"
        replay.write_text(json.dumps(events))
        report = parent / "report.json"
        command = [
            sys.executable, str(ROOT / "agent.py"), "--task", str(task),
            "--workdir", str(work), "--model", "stub", "--replay", str(replay),
            "--output", str(report),
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert process.returncode == 0, process.stderr
        assert process.stdout == "add.py fixed; tiny test passed.\n"
        result = json.loads(report.read_text())
        assert result["status"] == "succeeded"
        assert len(result["requests"]) == 5
        assert len(result["transport_attempts"]) == 6
        assert result["tools"][2]["result"] == "exit 0\ntiny test passed\n"
        assert result["tools"][3]["result"].startswith("UNCHANGED:")
        assert "a + b" in (work / "add.py").read_text()
        # The same entry point returns nonzero and no answer on truncation.
        replay.write_text(json.dumps([reply("partial", "length")]))
        process = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert process.returncode == 1
        assert process.stdout == ""
        assert json.loads(report.read_text())["final"] is None


def main() -> None:
    tests: list[Callable[[], None]] = [
        termination, independent_budgets, streaming_reasoning_budget, streaming_answer_budget,
        streaming_cumulative_budget, streaming_retry_keeps_reasoning_charge,
        exa_streaming_wire_budget, exa_streaming_wire_completion,
        raw_stream_deadline, silent_stream_remaining_timeout, stream_timeouts_share_retry_budget,
        watchdog_interrupts_partial_line,
        localhost_request_deadline,
        transport_retries, retry_budget_and_status,
        concurrency_limit, separate_times, sandbox_boundary, duplicate_guard,
        cli_end_to_end,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED (offline, including CLI end-to-end)")


if __name__ == "__main__":
    main()
