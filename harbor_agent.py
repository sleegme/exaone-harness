#!/usr/bin/env python3
"""Harbor adapter for the EXAONE coding agent.

Import as ``harbor_agent:ExaHarnessAgent`` with this file's directory on
``PYTHONPATH``. The adapter uploads ``agent.py``/``guide.txt`` into the task
container and runs ``python3 agent.py --model exa --no-bwrap`` there (the
container already provides what bubblewrap provides on a host; the reasoning
budget is raised to 258048 so long multi-turn runs are bounded by the task
timeout instead of the tiny standalone default).
``EXAONE_API_KEY`` is read from the harbor process environment and passed only
to the container exec call, never to job configs, logs, or files.
"""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

REMOTE_DIR = "/logs/agent/exa-harness"
REASONING_BUDGET = 258048
RETRY_BUDGET_SEC = 900


class ExaHarnessAgent(BaseInstalledAgent):
    @classmethod
    def name(cls) -> str:
        return "exa-harness"

    async def install(self, environment: BaseEnvironment) -> None:
        check = await environment.exec(
            command=(
                'command -v python3 >/dev/null 2>&1 && '
                'python3 -c "import ssl; assert ssl.create_default_context().get_ca_certs()"'
            ),
            user="root",
        )
        if check.return_code != 0:
            await self.ensure_system_dependencies(environment, ("python3", "ca_certificates"))

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        key = os.environ.get("EXAONE_API_KEY", "")
        if not key:
            raise RuntimeError(
                "EXAONE_API_KEY must be set in the harbor process environment"
            )

        here = Path(__file__).resolve().parent
        await self.exec_as_root(environment, f"mkdir -p {shlex.quote(REMOTE_DIR)}")
        for name in ("agent.py", "guide.txt"):
            await environment.upload_file(here / name, f"{REMOTE_DIR}/{name}")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as task_file:
            task_file.write(instruction)
            task_path = task_file.name
        try:
            await environment.upload_file(task_path, f"{REMOTE_DIR}/task.txt")
        finally:
            os.unlink(task_path)
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                f"chown -R {shlex.quote(str(environment.default_user))} "
                f"{shlex.quote(REMOTE_DIR)}",
            )

        command = (
            f"python3 {REMOTE_DIR}/agent.py"
            f" --task {REMOTE_DIR}/task.txt"
            ' --workdir "$(pwd)"'
            " --model exa --no-bwrap"
            f" --reasoning-budget {REASONING_BUDGET} --retry-budget {RETRY_BUDGET_SEC}"
            f" --output {REMOTE_DIR}/result.json"
        )
        await self._exec(environment, command, env={"EXAONE_API_KEY": key})
