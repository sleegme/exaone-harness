"""Offline contract tests. Run with python3 tests.py; no test dependencies."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from agent import (
    Budgets, Document, Executor, JSON, ReplayClient, RequestFailed, RetryPolicy,
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
        termination, independent_budgets, transport_retries, retry_budget_and_status,
        concurrency_limit, separate_times, sandbox_boundary, duplicate_guard,
        cli_end_to_end,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED (offline, including CLI end-to-end)")


if __name__ == "__main__":
    main()
