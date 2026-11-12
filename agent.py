#!/usr/bin/env python3
"""A synchronous coding agent with executor-owned termination and isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Protocol, TypeAlias

JSON: TypeAlias = str | int | float | bool | None | list["JSON"] | dict[str, "JSON"]
Document: TypeAlias = dict[str, JSON]
PROTECTED = (Path("/tmp/exa-harness"), Path("/tmp/exa-bench"))
IGNORED_STATE = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache"})


class ProtocolError(ValueError):
    """An invalid model or tool protocol value."""


class TransportError(RuntimeError):
    """A failed HTTP attempt, without credentials or response bodies."""

    def __init__(self, status: int | None, retry_after: str | None = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"transport status={status}")


class RequestFailed(RuntimeError):
    """A logical request could not obtain a usable transport response."""

    def __init__(self, request_id: str, reason: str):
        self.request_id = request_id
        self.reason = reason
        super().__init__(reason)


def mapping(value: JSON) -> Document:
    if not isinstance(value, dict):
        raise ProtocolError("expected JSON object")
    return value


def text(value: JSON) -> str:
    if not isinstance(value, str):
        raise ProtocolError("expected string")
    return value


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class Budgets:
    reasoning_tokens: int = 4096
    final_tokens: int = 4096

    def __post_init__(self) -> None:
        if self.reasoning_tokens < 0 or self.final_tokens < 1:
            raise ProtocolError("reasoning budget >= 0 and final budget >= 1 required")

    def parameters(self, separate: bool) -> Document:
        # Only clients declaring independent caps receive these fields.
        if separate:
            return {
                "reasoning_max_tokens": self.reasoning_tokens,
                "max_tokens": self.final_tokens,
            }
        # The supplied EXAONE harness has no verified numeric reasoning-cap field.
        return {
            "max_tokens": self.reasoning_tokens + self.final_tokens,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": True},
        }


class Client(Protocol):
    separate_budgets: bool

    def complete(self, payload: Document, timeout: float) -> Document: ...


class ExaClient:
    """OpenAI-compatible wire adapter; credentials are read only when selected."""

    separate_budgets = False

    def __init__(self, key: str):
        self.key = key

    def complete(self, payload: Document, timeout: float) -> Document:
        request = urllib.request.Request(
            "https://api.exaone.ai/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return mapping(json.loads(response.read()))
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After")
            status = error.code
            error.close()
            raise TransportError(status, retry_after) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise TransportError(None) from None


class ReplayClient:
    """Offline wire replay: response documents or {error:{status,retry_after}}."""

    separate_budgets = True

    def __init__(self, events: list[JSON]):
        self.events = iter(events)

    def complete(self, payload: Document, timeout: float) -> Document:
        del payload, timeout
        try:
            event = mapping(next(self.events))
        except StopIteration:
            raise ProtocolError("replay exhausted before final response") from None
        if "error" in event:
            error = mapping(event["error"])
            status = error.get("status")
            if status is not None and not isinstance(status, int):
                raise ProtocolError("error status must be integer or null")
            retry_after = error.get("retry_after")
            raise TransportError(status, None if retry_after is None else str(retry_after))
        return event


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    budget_seconds: float = 120
    timeout_seconds: float = 60
    max_attempts: int = 6
    concurrency: int = 1

    def __post_init__(self) -> None:
        if min(self.budget_seconds, self.timeout_seconds, self.max_attempts, self.concurrency) <= 0:
            raise ProtocolError("retry settings must all be positive")


class Executor:
    """Owns transport retries and a shared limit across all users of this executor."""

    def __init__(
        self,
        client: Client,
        policy: RetryPolicy = RetryPolicy(),
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.client = client
        self.policy = policy
        self.clock, self.sleep, self.wall_clock = clock, sleep, wall_clock
        self.limit = threading.BoundedSemaphore(policy.concurrency)
        self.attempts: list[Document] = []
        self.lock = threading.Lock()

    def request(self, payload: Document, request_id: str) -> Document:
        deadline = self.clock() + self.policy.budget_seconds
        if not self.limit.acquire(timeout=self.policy.budget_seconds):
            raise RequestFailed(request_id, "concurrency_wait_budget")
        try:
            for attempt in range(1, self.policy.max_attempts + 1):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise RequestFailed(request_id, "retry_time_budget")
                started, start_wall = self.clock(), utc()
                record: Document = {
                    "request_id": request_id,
                    "attempt_id": f"{request_id}:{attempt}",
                    "attempt": attempt,
                    "retransmission": attempt > 1,
                    "started_at": start_wall,
                    "status": None,
                    "wait_seconds": 0.0,
                }
                error: TransportError | None = None
                response: Document | None = None
                try:
                    response = self.client.complete(
                        payload, min(remaining, self.policy.timeout_seconds)
                    )
                    record["status"] = 200
                except TransportError as caught:
                    error = caught
                    record["status"] = caught.status
                finally:
                    record["ended_at"] = utc()
                    record["model_seconds"] = self.clock() - started
                    with self.lock:
                        self.attempts.append(record)
                if self.clock() > deadline:
                    raise RequestFailed(request_id, "retry_time_budget")
                if response is not None:
                    return response
                if error is None:
                    raise ProtocolError("transport returned neither response nor error")
                retryable = error.status is None or error.status == 429 or 500 <= error.status <= 599
                if not retryable or attempt == self.policy.max_attempts:
                    raise RequestFailed(request_id, "transport_exhausted") from error
                wait = min(2 ** (attempt - 1), 30)
                if error.retry_after is not None:
                    record["retry_after"] = error.retry_after
                    try:
                        wait = max(0.0, float(error.retry_after))
                    except ValueError:
                        try:
                            date = parsedate_to_datetime(error.retry_after)
                            wait = max(0.0, date.timestamp() - self.wall_clock())
                        except (ValueError, TypeError, OverflowError):
                            record["invalid_retry_after"] = True
                record["scheduled_wait_seconds"] = wait
                if wait >= deadline - self.clock():
                    raise RequestFailed(request_id, "retry_time_budget")
                before_wait = self.clock()
                self.sleep(wait)
                record["wait_seconds"] = self.clock() - before_wait
            raise RequestFailed(request_id, "transport_exhausted")
        finally:
            self.limit.release()


def tool_spec(name: str, description: str, properties: Document, required: list[str]) -> Document:
    return {
        "type": "function",
        "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


TOOLS = [
    tool_spec("list_dir", "List task directory", {"path": {"type": "string"}}, ["path"]),
    tool_spec("read_file", "Read up to 400 task file lines", {
        "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"},
    }, ["path"]),
    tool_spec("grep", "Regex search inside task files", {
        "path": {"type": "string"}, "pattern": {"type": "string"},
    }, ["path", "pattern"]),
    tool_spec("write_file", "Write task file", {
        "path": {"type": "string"}, "content": {"type": "string"},
    }, ["path", "content"]),
    tool_spec("run", "Run shell in /work, sequentially, no network, 60s timeout. "
              "An identical command with unchanged task files returns its previous summary.", {
                  "cmd": {"type": "string"},
              }, ["cmd"]),
]


class Sandbox:
    """No host root mount: only runtimes read-only and task files read/write."""

    def __init__(self, workdir: Path):
        self.root = workdir.resolve(strict=True)
        if not self.root.is_dir():
            raise ProtocolError("workdir must be a directory")
        for protected in PROTECTED:
            if self.root.is_relative_to(protected) or protected.is_relative_to(self.root):
                raise PermissionError("workdir overlaps a protected benchmark directory")
        if not shutil.which("bwrap"):
            raise ProtocolError("bubblewrap is required; shell isolation cannot be disabled")
        self.checks: dict[str, str] = {}

    def path(self, value: str) -> Path:
        # Shell sees /work; Python tools accept that spelling or task-relative paths.
        candidate = Path(value)
        if value == "/work" or value.startswith("/work/"):
            candidate = Path(value).relative_to("/work")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError("path outside task workdir")
        return resolved

    def state(self) -> str:
        """Hash names, contents, executable mode and symlink targets, not mtimes."""
        digest = hashlib.sha256()
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_STATE)
            for name in sorted(dirs + files):
                path = Path(directory) / name
                digest.update(str(path.relative_to(self.root)).encode() + b"\0")
                digest.update(str(path.lstat().st_mode).encode() + b"\0")
                if path.is_symlink():
                    digest.update(os.readlink(path).encode())
                elif path.is_file():
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(65536), b""):
                            digest.update(block)
                digest.update(b"\0")
        return digest.hexdigest()

    def shell(self, command: str) -> str:
        fingerprint = hashlib.sha256((command + "\0" + self.state()).encode()).hexdigest()
        if fingerprint in self.checks:
            return "UNCHANGED: identical check blocked; previous result: " + self.checks[fingerprint]
        argv = [
            "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--remount-ro", "/tmp",
            "--bind", str(self.root), "/work", "--chdir", "/work",
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", "/work", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "sh", "-c", command,
        ]
        try:
            result = subprocess.run(
                argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=60, env={"PATH": "/usr/bin:/bin"},
            )
            summary = f"exit {result.returncode}\n{(result.stdout + result.stderr)[-8000:]}"
        except subprocess.TimeoutExpired:
            summary = "ERROR: shell timed out after 60 seconds"
        self.checks[fingerprint] = summary[:1200]
        return summary

    def execute(self, name: str, arguments: Document) -> str:
        try:
            match name:
                case "run":
                    return self.shell(text(arguments["cmd"]))
                case "list_dir":
                    return "\n".join(sorted(p.name for p in self.path(text(arguments["path"])).iterdir()))
                case "read_file":
                    lines = self.path(text(arguments["path"])).read_text().splitlines()
                    start = max(1, int(arguments.get("start", 1)))
                    end = min(len(lines), int(arguments.get("end", start + 399)), start + 399)
                    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
                case "write_file":
                    target = self.path(text(arguments["path"]))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text(arguments["content"]))
                    return "ok"
                case "grep":
                    base = self.path(text(arguments["path"]))
                    pattern = re.compile(text(arguments["pattern"]))
                    paths = sorted(base.rglob("*")) if base.is_dir() else [base]
                    hits = []
                    for candidate in paths:
                        candidate = self.path(str(candidate))
                        if candidate.is_file():
                            for index, line in enumerate(candidate.read_text().splitlines(), 1):
                                if pattern.search(line):
                                    hits.append(f"{candidate.relative_to(self.root)}:{index}: {line}")
                    return "\n".join(hits)[:20000] or "(no matches)"
                case _:
                    raise ProtocolError(f"unknown tool: {name}")
        except (OSError, ValueError, TypeError, KeyError, re.error) as error:
            return f"ERROR: {error}"


@dataclass(frozen=True, slots=True)
class Reply:
    finish_reason: str
    message: Document
    usage: Document
    calls: list[Document]

    @classmethod
    def parse(cls, response: Document) -> Reply:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProtocolError("missing choice")
        choice = mapping(choices[0])
        message = mapping(choice.get("message"))
        role = text(message.get("role"))
        if role != "assistant":
            raise ProtocolError(f"expected assistant role, got {role}")
        content = message.get("content")
        if content is not None:
            text(content)
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ProtocolError("tool_calls must be an array")
        return cls(
            text(choice.get("finish_reason")), message,
            mapping(response.get("usage") or {}), [mapping(call) for call in calls],
        )


def run_agent(
    task: str, sandbox: Sandbox, executor: Executor, guide: str,
    *, max_turns: int = 25, budgets: Budgets = Budgets(),
) -> Document:
    """Only a complete assistant stop with nonempty final content can succeed."""
    if max_turns < 1:
        raise ProtocolError("max_turns must be positive")
    messages: list[JSON] = [
        {"role": "system", "content": guide + "\nTools use paths relative to /work. "
         "Shell calls run sequentially. Read/write access is limited to the task folder."},
        {"role": "user", "content": task},
    ]
    records: list[JSON] = []
    tool_records: list[JSON] = []
    started = executor.clock()
    outcome: Document = {
        "status": "failed", "failure": "turn_limit", "final": None,
        "budget": {
            "reasoning_tokens": budgets.reasoning_tokens,
            "final_tokens": budgets.final_tokens,
            "independent_caps_supported": executor.client.separate_budgets,
            "mode": "independent" if executor.client.separate_budgets else "shared_total_low_effort",
        },
    }
    for turn in range(1, max_turns + 1):
        request_id = uuid.uuid4().hex
        record: Document = {
            "request_id": request_id, "turn": turn, "started_at": utc(),
            "finish_reason": None, "usage": None, "role": None,
            "has_tools": False, "failure": None,
        }
        records.append(record)
        try:
            payload: Document = {
                "model": "k-exaone_v2.5", "messages": list(messages), "tools": list(TOOLS),
                **budgets.parameters(executor.client.separate_budgets),
            }
            response = executor.request(payload, request_id)
            # Preserve the raw response facts even when the message is malformed.
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                choice = mapping(choices[0])
                raw_message = mapping(choice.get("message"))
                record.update(
                    finish_reason=choice.get("finish_reason"),
                    usage=response.get("usage"), role=raw_message.get("role"),
                    has_tools=bool(raw_message.get("tool_calls")),
                )
            reply = Reply.parse(response)
            record.update(
                finish_reason=reply.finish_reason, usage=reply.usage,
                role=reply.message["role"], has_tools=bool(reply.calls),
            )
        except RequestFailed as error:
            record["failure"] = outcome["failure"] = error.reason
            break
        except (ProtocolError, json.JSONDecodeError) as error:
            record["failure"] = outcome["failure"] = f"protocol: {error}"
            break
        finally:
            record["ended_at"] = utc()
        # Truncated calls are never executed; partial content is never published.
        if reply.finish_reason not in {"stop", "tool_calls"}:
            record["failure"] = outcome["failure"] = reply.finish_reason
            break
        if not reply.calls:
            content = (reply.message.get("content") or "").strip()
            if reply.finish_reason != "stop" or not content:
                record["failure"] = outcome["failure"] = "empty_reply"
                break
            # Inline reasoning is not a final answer, even if finish_reason is stop.
            if "<think>" in content or "</think>" in content:
                record["failure"] = outcome["failure"] = "reasoning_in_final"
                break
            outcome.update(status="succeeded", failure=None, final=content)
            messages.append(reply.message)
            break
        messages.append(reply.message)
        for call in reply.calls:
            before, start_wall = executor.clock(), utc()
            try:
                function = mapping(call.get("function"))
                name = text(function.get("name"))
                arguments = mapping(json.loads(text(function.get("arguments"))))
                result = sandbox.execute(name, arguments)
                call_id = text(call.get("id"))
            except (ProtocolError, json.JSONDecodeError) as error:
                record["failure"] = outcome["failure"] = f"tool_protocol: {error}"
                break
            tool_records.append({
                "request_id": request_id, "tool_call_id": call_id, "name": name,
                "started_at": start_wall, "ended_at": utc(),
                "tool_seconds": executor.clock() - before, "result": result,
            })
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        if record["failure"] is not None:
            break
    if outcome["failure"] == "turn_limit":
        mapping(records[-1])["failure"] = "turn_limit"
    request_ids = {mapping(record)["request_id"] for record in records}
    attempts = [a for a in executor.attempts if a["request_id"] in request_ids]
    model_seconds = sum(float(a["model_seconds"]) for a in attempts)
    tool_seconds = sum(float(mapping(t)["tool_seconds"]) for t in tool_records)
    outcome.update(
        requests=records, transport_attempts=list(attempts),
        tools=tool_records, transcript=messages,
        model_seconds=model_seconds, tool_seconds=tool_seconds,
        retry_wait_seconds=sum(float(a["wait_seconds"]) for a in attempts),
        wall_seconds=executor.clock() - started,
    )
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--model", choices=["exa", "stub"], default="exa")
    parser.add_argument("--replay", type=Path, help="JSON array of wire responses/errors; stub only")
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--reasoning-budget", type=int, default=4096)
    parser.add_argument("--final-budget", type=int, default=4096)
    parser.add_argument("--retry-budget", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, help="JSON report outside the task folder")
    args = parser.parse_args()
    client: Client
    match args.model:
        case "stub":
            if args.replay is None:
                parser.error("--model stub requires --replay")
            events = json.loads(args.replay.read_text())
            if not isinstance(events, list):
                parser.error("--replay must contain a JSON array")
            client = ReplayClient(events)
        case "exa":
            key = os.environ.get("EXAONE_API_KEY")
            if not key:
                parser.error("EXAONE_API_KEY is required for --model exa")
            client = ExaClient(key)
    sandbox = Sandbox(args.workdir)
    output = args.output or Path(__file__).parent / "logs" / f"{uuid.uuid4().hex}.json"
    output = output.resolve()
    if output.is_relative_to(sandbox.root):
        parser.error("--output must be outside the task folder")
    for protected in PROTECTED:
        if output.is_relative_to(protected):
            parser.error("--output cannot be in protected benchmark directories")
    result = run_agent(
        args.task.read_text(), sandbox,
        Executor(client, RetryPolicy(args.retry_budget, args.request_timeout, concurrency=args.concurrency)),
        Path(__file__).with_name("guide.txt").read_text(),
        max_turns=args.max_turns, budgets=Budgets(args.reasoning_budget, args.final_budget),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if result["status"] == "succeeded":
        print(result["final"])
        return 0
    print(f"FAILED: {result['failure']} (report: {output})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
