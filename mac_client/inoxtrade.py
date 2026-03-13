#!/usr/bin/env python3
"""INOXTRADE — Remote CLI for controlling the trading bot from Mac via SSH tunnel.

Usage:
    python inoxtrade.py setup      # configure VPS connection
    python inoxtrade.py connect    # live dashboard
    python inoxtrade.py status     # one-line status
    python inoxtrade.py start      # start bot on VPS
    python inoxtrade.py stop       # stop bot on VPS
    python inoxtrade.py trades     # today's trades table
    python inoxtrade.py logs       # tail bot logs (live)
"""

from __future__ import annotations

import json
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_DIR = Path.home() / ".inoxtrade"
CONFIG_FILE = CONFIG_DIR / "config.json"
API_PORT = 8765
LOCAL_PORT = 8765


def _load_config() -> Dict[str, str]:
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text())


def _save_config(cfg: Dict[str, str]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# SSH Tunnel via paramiko
# ---------------------------------------------------------------------------
class SSHTunnel:
    """Forward localhost:LOCAL_PORT -> VPS:API_PORT through SSH."""

    def __init__(self, host: str, user: str, key_path: str, remote_port: int = API_PORT):
        self.host = host
        self.user = user
        self.key_path = key_path
        self.remote_port = remote_port
        self._client = None
        self._transport = None
        self._forward_thread = None
        self._server = None

    def open(self) -> None:
        import paramiko
        import select
        import socket
        import socketserver

        key_file = Path(self.key_path).expanduser()
        if not key_file.exists():
            raise FileNotFoundError(f"SSH key not found: {key_file}")

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self._client.connect(
                hostname=self.host,
                username=self.user,
                key_filename=str(key_file),
                timeout=10,
            )
        except Exception as e:
            raise ConnectionError(
                f"SSH connection failed to {self.user}@{self.host}\n"
                f"  Error: {e}\n"
                f"  Verify: IP address, SSH key, user, and that VPS is reachable."
            ) from e

        self._transport = self._client.get_transport()
        remote_port = self.remote_port
        transport = self._transport

        class ForwardHandler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    chan = transport.open_channel(
                        "direct-tcpip",
                        ("127.0.0.1", remote_port),
                        self.request.getpeername(),
                    )
                except Exception:
                    return
                if chan is None:
                    return
                try:
                    while True:
                        r, _, _ = select.select([self.request, chan], [], [], 1.0)
                        if self.request in r:
                            data = self.request.recv(4096)
                            if not data:
                                break
                            chan.sendall(data)
                        if chan in r:
                            data = chan.recv(4096)
                            if not data:
                                break
                            self.request.sendall(data)
                except Exception:
                    pass
                finally:
                    chan.close()

        class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = ThreadedTCPServer(("127.0.0.1", LOCAL_PORT), ForwardHandler)
        self._forward_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._forward_thread.start()

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
        if self._client:
            self._client.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------
BASE_URL = f"http://127.0.0.1:{LOCAL_PORT}"


def _api_get(endpoint: str, params: dict = None) -> Any:
    import requests
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r.text
    except requests.ConnectionError:
        _print_api_unreachable()
        sys.exit(1)


def _api_post(endpoint: str) -> Any:
    import requests
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", timeout=15)
        return r.json()
    except requests.ConnectionError:
        _print_api_unreachable()
        sys.exit(1)


def _print_api_unreachable():
    from rich.console import Console
    c = Console(stderr=True)
    c.print("[bold red]API non disponible sur le VPS.[/]")
    c.print("Lance [bold]start_api.bat[/] sur le VPS, puis reessaie.")


def _check_api() -> bool:
    """Verify the API is responding."""
    import requests
    try:
        r = requests.get(f"{BASE_URL}/ping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup():
    """Interactive config setup."""
    from rich.console import Console
    c = Console()

    c.print("\n[bold cyan]INOXTRADE Setup[/]\n")
    c.print("Configure connection to your VPS.\n")

    cfg = _load_config()

    host = input(f"  VPS Host/IP [{cfg.get('host', '')}]: ").strip()
    if host:
        cfg["host"] = host

    user = input(f"  SSH User [{cfg.get('user', 'Administrator')}]: ").strip()
    cfg["user"] = user or cfg.get("user", "Administrator")

    default_key = cfg.get("ssh_key", "~/.ssh/id_rsa")
    key = input(f"  SSH Key Path [{default_key}]: ").strip()
    cfg["ssh_key"] = key or default_key

    if not cfg.get("host"):
        c.print("[bold red]Host is required.[/]")
        sys.exit(1)

    _save_config(cfg)
    c.print(f"\n[bold green]Config saved to {CONFIG_FILE}[/]\n")
    c.print(f"  Host    : {cfg['host']}")
    c.print(f"  User    : {cfg['user']}")
    c.print(f"  SSH Key : {cfg['ssh_key']}")
    c.print()


def cmd_status():
    """One-line status."""
    from rich.console import Console
    c = Console()

    data = _api_get("/status")
    bot = data["bot"]
    acct = data["account"]
    mkt = data["market"]
    trades = data["trades"]

    state = "[bold green]RUNNING[/]" if bot["running"] else "[bold red]STOPPED[/]"
    bal = f"${acct['balance']:,.2f}" if acct["balance"] else "N/A"
    pnl = acct["daily_pnl"] or 0
    pnl_s = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    pnl_style = "green" if pnl >= 0 else "red"
    bid = f"${mkt['bid']:,.2f}" if mkt["bid"] else "N/A"

    c.print(
        f"  INOXTRADE {state}  |  "
        f"Balance {bal}  |  "
        f"P&L [{pnl_style}]{pnl_s}[/]  |  "
        f"XAUUSD {bid}  |  "
        f"Trades: {trades['total_today']}  |  "
        f"{mkt['session']}"
    )


def cmd_start():
    """Start the bot on VPS."""
    from rich.console import Console
    c = Console()

    result = _api_post("/start")
    if result.get("success"):
        c.print(f"[bold green]Bot demarre[/] (PID {result.get('pid')})")
    else:
        c.print(f"[bold red]Echec[/]: {result.get('message')}")


def cmd_stop():
    """Stop the bot on VPS (with confirmation)."""
    from rich.console import Console
    c = Console()

    confirm = input("  Arreter le bot ? (y/n): ").strip().lower()
    if confirm != "y":
        c.print("[dim]Annule.[/]")
        return

    result = _api_post("/stop")
    if result.get("success"):
        c.print("[bold green]Bot arrete.[/]")
    else:
        c.print(f"[bold red]Echec[/]: {result.get('message')}")


def cmd_trades():
    """Display today's trades in a rich table."""
    from rich.console import Console
    from rich.table import Table
    c = Console()

    data = _api_get("/trades")
    trades = data.get("trades", [])

    table = Table(title="Trades du Jour", border_style="dim", expand=True)
    table.add_column("Heure", style="cyan", width=8)
    table.add_column("Dir", width=6)
    table.add_column("Entree", justify="right", width=10)
    table.add_column("Sortie", justify="right", width=10)
    table.add_column("SL", justify="right", width=10)
    table.add_column("Etat", width=8)
    table.add_column("P&L", justify="right", width=12)

    if not trades:
        table.add_row("--", "--", "--", "--", "--", "--", "--")
    else:
        for t in trades:
            hour = t["opened_at"][11:16] if t.get("opened_at") else "--"
            d = t.get("direction", "--")
            dir_style = "green" if d == "LONG" else "red"
            entry = f"${t['entry_price']:,.2f}" if t.get("entry_price") else "--"
            exit_p = f"${t['exit_price']:,.2f}" if t.get("exit_price") else "[cyan]OPEN[/]"
            sl = f"${t['stop_loss']:,.2f}" if t.get("stop_loss") else "--"
            state = t.get("state", "--")
            pnl = t.get("pnl")
            if pnl is not None:
                ps = "bold green" if pnl >= 0 else "bold red"
                sign = "+" if pnl >= 0 else ""
                pnl_str = f"[{ps}]{sign}${pnl:,.2f}[/]"
            else:
                pnl_str = "[dim]---[/]"

            table.add_row(hour, f"[{dir_style}]{d}[/]", entry, exit_p, sl, state, pnl_str)

    c.print(table)
    c.print(f"[dim]{data.get('count', 0)} trades | {data.get('timestamp', '')[11:19]} UTC[/]")


def cmd_logs():
    """Tail bot logs in real-time (poll every 2s)."""
    from rich.console import Console
    c = Console()

    c.print("[bold cyan]Bot Logs[/] (Ctrl+C to quit)\n")

    last_len = 0
    try:
        while True:
            text = _api_get("/logs", params={"lines": 50})
            lines = text.strip().split("\n") if isinstance(text, str) else []
            current_len = len(lines)

            if current_len != last_len or last_len == 0:
                c.clear()
                c.print("[bold cyan]Bot Logs[/] (Ctrl+C to quit)\n")
                for line in lines:
                    c.print(line)
                last_len = current_len

            time.sleep(2)
    except KeyboardInterrupt:
        c.print("\n[dim]Logs stopped.[/]")


def cmd_connect():
    """Live dashboard with 5s refresh."""
    from rich.align import Align
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HEADER = r"""
 ___ _   _  _____  _______ ____      _    ____  _____
|_ _| \ | |/ _ \ \/ /_   _|  _ \    / \  |  _ \| ____|
 | ||  \| | | | \  /  | | | |_) |  / _ \ | | | |  _|
 | || |\  | |_| /  \  | | |  _ <  / ___ \| |_| | |___
|___|_| \_|\___/_/\_\ |_| |_| \_\/_/   \_\____/|_____|
"""

    def build_layout():
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=11),
            Layout(name="body", size=10),
            Layout(name="trades"),
        )
        layout["body"].split_row(
            Layout(name="account", ratio=1),
            Layout(name="market", ratio=1),
            Layout(name="last_trade", ratio=1),
        )
        return layout

    def refresh(layout):
        try:
            status = _api_get("/status")
            trades_data = _api_get("/trades")
        except SystemExit:
            return

        bot = status.get("bot", {})
        acct = status.get("account", {})
        mkt = status.get("market", {})
        tr_summary = status.get("trades", {})
        trades = trades_data.get("trades", [])

        now = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")

        # -- Header --
        hdr = Text()
        hdr.append(HEADER.strip(), style="bold cyan")
        hdr.append("\n")
        hdr.append("Gold Algorithmic Trading System", style="bold white")
        hdr.append("  |  ", style="dim")
        hdr.append("XAUUSD", style="bold yellow")
        hdr.append(" . IC Markets ECN . MT5", style="white")
        hdr.append("  |  ", style="dim")
        hdr.append(now, style="bold green")
        hdr.append("  |  ", style="dim")
        state_txt = "RUNNING" if bot.get("running") else "STOPPED"
        state_sty = "bold green" if bot.get("running") else "bold red"
        hdr.append(state_txt, style=state_sty)
        layout["header"].update(Panel(Align.center(hdr), border_style="cyan"))

        # -- Account --
        g = Table.grid(padding=(0, 3))
        g.add_column(justify="right", style="dim", min_width=16)
        g.add_column(justify="left", min_width=20)

        bal = acct.get("balance")
        if not acct.get("mt5_connected"):
            g.add_row("Statut MT5", Text("MT5 DECONNECTE", style="bold red"))
            g.add_row("Balance", Text("---", style="dim"))
        else:
            g.add_row("Balance", Text(f"${bal:,.2f}", style="bold white"))

        pnl = acct.get("daily_pnl", 0) or 0
        pnl_pct = acct.get("daily_pnl_pct", 0) or 0
        ps = "bold green" if pnl >= 0 else "bold red"
        sign = "+" if pnl >= 0 else ""
        g.add_row("P&L jour", Text(f"{sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)", style=ps))

        dd = abs(pnl_pct) if pnl < 0 else 0.0
        dd_t = Text(f"{dd:.2f}%", style="bold red" if dd >= 3 else "white")
        dd_t.append(" / 5.00%", style="dim")
        g.add_row("Drawdown", dd_t)

        wr = tr_summary.get("win_rate", 0)
        g.add_row("WR jour", Text(f"{wr:.1f}% ({tr_summary.get('wins',0)}W/{tr_summary.get('losses',0)}L)", style="white"))

        layout["account"].update(Panel(g, title="[bold]Compte[/]", border_style="blue"))

        # -- Market --
        m = Table.grid(padding=(0, 3))
        m.add_column(justify="right", style="dim", min_width=16)
        m.add_column(justify="left", min_width=20)

        bias = mkt.get("daily_bias", "N/A")
        bias_s = {"BULLISH": "bold green", "BEARISH": "bold red", "NEUTRAL": "bold yellow"}.get(bias, "dim")
        m.add_row("Daily Bias", Text(bias, style=bias_s))

        atr = mkt.get("atr_m1")
        m.add_row("ATR M1", Text(f"{atr:.4f}" if atr else "---", style="bold green" if atr and atr >= 0.4 else "dim"))

        sess = mkt.get("session", "---")
        ss = {"London": "bold cyan", "New York": "bold magenta"}.get(sess, "dim italic")
        m.add_row("Session", Text(sess, style=ss))

        bid = mkt.get("bid")
        m.add_row("XAUUSD", Text(f"${bid:,.2f}" if bid else "---", style="bold yellow"))

        layout["market"].update(Panel(m, title="[bold]Marche[/]", border_style="magenta"))

        # -- Last trade --
        if trades:
            last = trades[-1]
            lt = Table.grid(padding=(0, 3))
            lt.add_column(justify="right", style="dim", min_width=14)
            lt.add_column(justify="left", min_width=18)

            d = last.get("direction", "---")
            lt.add_row("Direction", Text(d, style="bold green" if d == "LONG" else "bold red"))
            lt.add_row("Entree", Text(f"${last['entry_price']:,.2f}" if last.get("entry_price") else "---"))
            ep = last.get("exit_price")
            lt.add_row("Sortie", Text(f"${ep:,.2f}" if ep else "EN COURS", style="white" if ep else "bold cyan"))
            lp = last.get("pnl")
            if lp is not None:
                lt.add_row("P&L", Text(f"{'+'if lp>=0 else ''}${lp:,.2f}", style="bold green" if lp >= 0 else "bold red"))
            else:
                lt.add_row("P&L", Text("---", style="dim"))
            lt.add_row("Etat", Text(last.get("state", "---")))
            layout["last_trade"].update(Panel(lt, title="[bold]Dernier Trade[/]", border_style="yellow"))
        else:
            layout["last_trade"].update(
                Panel(Align.center(Text("Aucun trade aujourd'hui", style="dim italic")),
                      title="[bold]Dernier Trade[/]", border_style="yellow")
            )

        # -- Trades table --
        table = Table(show_header=True, header_style="bold white", border_style="dim", expand=True)
        table.add_column("Heure", style="cyan", width=8)
        table.add_column("Dir", width=6)
        table.add_column("Entree", justify="right", width=10)
        table.add_column("Sortie", justify="right", width=10)
        table.add_column("Etat", width=8)
        table.add_column("P&L", justify="right", width=12)

        if not trades:
            table.add_row("--", "--", "--", "--", "--", "--")
        else:
            for i, t in enumerate(trades):
                h = t["opened_at"][11:16] if t.get("opened_at") else "--"
                d = t.get("direction", "--")
                ds = "green" if d == "LONG" else "red"
                en = f"${t['entry_price']:,.2f}" if t.get("entry_price") else "--"
                ex = f"${t['exit_price']:,.2f}" if t.get("exit_price") else "[cyan]OPEN[/]"
                st = t.get("state", "--")
                p = t.get("pnl")
                if p is not None:
                    pst = "bold green" if p >= 0 else "bold red"
                    ps = f"[{pst}]{'+'if p>=0 else ''}${p:,.2f}[/]"
                else:
                    ps = "[dim]---[/]"
                sty = "bold on grey23" if i == len(trades) - 1 and st in ("OPEN", "TP1_HIT", "TP2_HIT") else ""
                table.add_row(h, f"[{ds}]{d}[/]", en, ex, st, ps, style=sty)

        layout["trades"].update(Panel(table, title="[bold]Trades du Jour[/]", border_style="green"))

    console = Console()
    layout = build_layout()

    try:
        with Live(layout, console=console, refresh_per_second=1, screen=True):
            while True:
                refresh(layout)
                time.sleep(5)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Dashboard stopped.[/]")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
COMMANDS = {
    "setup": cmd_setup,
    "connect": cmd_connect,
    "status": cmd_status,
    "start": cmd_start,
    "stop": cmd_stop,
    "trades": cmd_trades,
    "logs": cmd_logs,
}


def main():
    from rich.console import Console
    c = Console()

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        c.print("\n[bold cyan]INOXTRADE[/] — Remote Bot Control\n")
        c.print("Usage:  python inoxtrade.py <command>\n")
        c.print("Commands:")
        c.print("  [bold]setup[/]     Configure VPS connection")
        c.print("  [bold]connect[/]   Live dashboard")
        c.print("  [bold]status[/]    One-line status")
        c.print("  [bold]start[/]     Start bot on VPS")
        c.print("  [bold]stop[/]      Stop bot on VPS")
        c.print("  [bold]trades[/]    Today's trades table")
        c.print("  [bold]logs[/]      Tail bot logs (live)")
        c.print()
        sys.exit(0)

    command = sys.argv[1]

    # setup doesn't need SSH
    if command == "setup":
        COMMANDS[command]()
        return

    # All other commands need SSH tunnel
    cfg = _load_config()
    if not cfg.get("host"):
        c.print("[bold red]VPS non configure.[/]  Lance d'abord:  python inoxtrade.py setup")
        sys.exit(1)

    c.print(f"[dim]Connexion SSH a {cfg['user']}@{cfg['host']}...[/]")

    try:
        with SSHTunnel(
            host=cfg["host"],
            user=cfg["user"],
            key_path=cfg.get("ssh_key", "~/.ssh/id_rsa"),
        ):
            # Brief pause for tunnel to stabilize
            time.sleep(0.5)

            if not _check_api():
                _print_api_unreachable()
                sys.exit(1)

            c.print("[dim]Connecte.[/]\n")
            COMMANDS[command]()

    except FileNotFoundError as e:
        c.print(f"[bold red]{e}[/]")
        sys.exit(1)
    except ConnectionError as e:
        c.print(f"[bold red]{e}[/]")
        sys.exit(1)
    except KeyboardInterrupt:
        c.print("\n[dim]Deconnecte.[/]")


if __name__ == "__main__":
    main()
