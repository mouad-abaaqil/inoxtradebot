#!/usr/bin/env python
"""Register INOXTRADE API server in Windows Task Scheduler.

Creates a task that runs start_api.bat at user logon so the API
is always available after VPS reboot.

Usage:
    python scripts/setup_autostart.py          # create the task
    python scripts/setup_autostart.py --remove  # remove the task
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TASK_NAME = "INOXTRADE_API"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAT_PATH = PROJECT_ROOT / "start_api.bat"


def create_task() -> None:
    """Create a Windows Task Scheduler task to launch the API at logon."""
    if not BAT_PATH.exists():
        print(f"ERROR: {BAT_PATH} not found")
        sys.exit(1)

    # Ensure logs directory exists
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # Build schtasks command
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", f'"{BAT_PATH}"',
        "/SC", "ONLOGON",
        "/RL", "HIGHEST",
        "/F",  # force overwrite if exists
    ]

    print(f"Creating scheduled task '{TASK_NAME}'...")
    print(f"  Trigger : at user logon")
    print(f"  Action  : {BAT_PATH}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' created successfully.")
        print()
        print("Verify with:")
        print(f"  schtasks /Query /TN {TASK_NAME}")
        print()
        print("To test immediately:")
        print(f"  schtasks /Run /TN {TASK_NAME}")
    else:
        print(f"ERROR: schtasks failed (exit code {result.returncode})")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(1)


def remove_task() -> None:
    """Remove the scheduled task."""
    cmd = ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]

    print(f"Removing scheduled task '{TASK_NAME}'...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' removed.")
    else:
        print(f"ERROR: {result.stderr.strip() or result.stdout.strip()}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Setup INOXTRADE API autostart")
    parser.add_argument("--remove", action="store_true", help="Remove the scheduled task")
    args = parser.parse_args()

    if args.remove:
        remove_task()
    else:
        create_task()


if __name__ == "__main__":
    main()
