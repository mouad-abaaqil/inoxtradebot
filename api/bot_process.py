"""Bot process manager — start, stop, status of main.py subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = PROJECT_ROOT / "logs" / "bot.pid"
LOG_FILE = PROJECT_ROOT / "logs" / "bot.log"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
PYTHON = sys.executable


class BotProcess:
    """Manages the lifecycle of the main.py trading bot process."""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def is_running() -> bool:
        """Check if the bot process is alive."""
        pid = BotProcess._read_pid()
        if pid is None:
            return False
        return BotProcess._pid_alive(pid)

    @staticmethod
    def get_pid() -> Optional[int]:
        """Return the bot PID if running, else None."""
        pid = BotProcess._read_pid()
        if pid is not None and BotProcess._pid_alive(pid):
            return pid
        return None

    @staticmethod
    def status() -> dict:
        """Return a status dict for the API."""
        pid = BotProcess.get_pid()
        running = pid is not None
        info: dict = {
            "running": running,
            "pid": pid,
            "uptime_seconds": None,
        }
        if running and pid is not None:
            try:
                proc = psutil.Process(pid)
                info["uptime_seconds"] = round(time.time() - proc.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return info

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    @staticmethod
    def start() -> dict:
        """Start main.py as a detached subprocess.

        Returns a dict with success status and message.
        """
        if BotProcess.is_running():
            return {"success": False, "message": "Bot already running", "pid": BotProcess.get_pid()}

        # Ensure logs directory exists
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Open log file for stdout/stderr redirect
        log_fh = open(LOG_FILE, "a", encoding="utf-8")

        # Launch main.py detached
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008

        proc = subprocess.Popen(
            [PYTHON, str(MAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        )

        # Write PID file
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")

        # Give the process a moment to crash or settle
        time.sleep(1.0)
        if proc.poll() is not None:
            # Process already exited — startup failure
            PID_FILE.unlink(missing_ok=True)
            return {"success": False, "message": f"Bot crashed on startup (exit code {proc.returncode})"}

        return {"success": True, "message": "Bot started", "pid": proc.pid}

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    @staticmethod
    def stop() -> dict:
        """Stop the bot process gracefully (SIGTERM, then kill after 5s).

        Returns a dict with success status and message.
        """
        pid = BotProcess.get_pid()
        if pid is None:
            PID_FILE.unlink(missing_ok=True)
            return {"success": False, "message": "Bot is not running"}

        try:
            proc = psutil.Process(pid)

            # Graceful termination
            proc.terminate()

            # Wait up to 5 seconds for clean exit
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill
                proc.kill()
                proc.wait(timeout=3)

        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            # Try OS-level kill as fallback
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
            except (ProcessLookupError, OSError):
                pass

        # Cleanup PID file
        PID_FILE.unlink(missing_ok=True)

        # Verify it's actually stopped
        if BotProcess._pid_alive(pid):
            return {"success": False, "message": f"Failed to stop process {pid}"}

        return {"success": True, "message": "Bot stopped"}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pid() -> Optional[int]:
        """Read PID from the pid file. Returns None if missing or invalid."""
        if not PID_FILE.exists():
            return None
        try:
            text = PID_FILE.read_text(encoding="utf-8").strip()
            return int(text)
        except (ValueError, OSError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Check if a process with this PID exists and is a Python process."""
        try:
            proc = psutil.Process(pid)
            # Verify it's actually our bot (not a recycled PID)
            cmdline = " ".join(proc.cmdline()).lower()
            return "main.py" in cmdline or "python" in cmdline
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    # ------------------------------------------------------------------
    # Log reading
    # ------------------------------------------------------------------

    @staticmethod
    def tail_log(lines: int = 50) -> str:
        """Return the last N lines of the bot log file."""
        if not LOG_FILE.exists():
            return "(no log file)"
        try:
            all_lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return "\n".join(tail)
        except OSError:
            return "(error reading log)"
