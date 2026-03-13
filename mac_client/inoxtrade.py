#!/usr/bin/env python3
"""INOXTRADE — Remote CLI for controlling the trading bot from Mac via SSH tunnel.

Usage:
    python inoxtrade.py setup      # configure VPS connection
    python inoxtrade.py connect    # live dashboard
    python inoxtrade.py menu       # interactive menu
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


def _api_post(endpoint: str, json_body: dict | None = None) -> Any:
    import requests
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", json=json_body, timeout=15)
        return r.json()
    except requests.ConnectionError:
        _print_api_unreachable()
        sys.exit(1)


def _api_post_safe(endpoint: str, json_body: dict | None = None) -> Any:
    """Like _api_post but returns None on error."""
    import requests
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", json=json_body, timeout=15)
        return r.json()
    except Exception:
        return None


def _api_get_safe(endpoint: str, params: dict = None) -> Any:
    """Like _api_get but returns None instead of exiting on error."""
    import requests
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=5)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r.text
    except Exception:
        return None


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
    """Live terminal dashboard — exact visual spec."""
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    def _make_ascii_art() -> Text:
        """Build the INOXTRADE ASCII banner with only X in red, rest bright white."""
        raw_lines = [
            " ___ _   _  _____  _______ ____      _    ____  _____",
            "|_ _| \\ | |/ _ \\ \\/ /_   _|  _ \\    / \\  |  _ \\| ____|",
            " | ||  \\| | | | \\  /  | | | |_) |  / _ \\ | | | |  _|",
            " | || |\\  | |_| /  \\  | | |  _ <  / ___ \\| |_| | |___",
            "|___|_| \\_|\\___/_/\\_\\ |_| |_| \\_\\/_/   \\_\\____/|_____|",
        ]
        t = Text()
        for i, line in enumerate(raw_lines):
            for j, ch in enumerate(line):
                is_x = False
                if i == 1 and 13 <= j <= 15:
                    is_x = True
                elif i == 2 and 13 <= j <= 17:
                    is_x = True
                elif i == 3 and 13 <= j <= 17:
                    is_x = True
                elif i == 4 and 13 <= j <= 17:
                    is_x = True
                t.append(ch, style="bold red" if is_x else "bold bright_white")
            t.append("\n")
        return t

    def _disconnected_text() -> Text:
        return Text("API DECONNECTEE", style="bold red")

    def _build_dashboard(status: dict | None, trades_data: dict | None) -> Group:
        api_ok = status is not None
        now_utc = datetime.utcnow().strftime("%H:%M:%S UTC")

        bot = status.get("bot", {}) if api_ok else {}
        acct = status.get("account", {}) if api_ok else {}
        mkt = status.get("market", {}) if api_ok else {}
        tr_summary = status.get("trades", {}) if api_ok else {}
        trades = trades_data.get("trades", []) if trades_data else []

        parts = []

        # ── 1. HEADER ──────────────────────────────────────────
        header_txt = _make_ascii_art()
        subtitle = Text()
        subtitle.append("Gold Algorithmic Trading System — XAUUSD", style="dim")
        subtitle.append(" . IC Markets ECN . MT5", style="dim")
        subtitle.append("    ", style="dim")
        subtitle.append(now_utc, style="bold white")

        header_group = Text()
        header_group.append_text(header_txt)
        header_group.append_text(subtitle)
        parts.append(Align.center(header_group))

        # ── 2. METRIQUES (4 panels) ────────────────────────────
        if api_ok:
            bal = acct.get("balance")
            bal_txt = Text(f"${bal:,.2f}" if bal else "---", style="bold white")
            bal_sub = Text("[IC Markets]", style="dim")

            pnl = acct.get("daily_pnl", 0) or 0
            pnl_sign = "+" if pnl >= 0 else ""
            pnl_style = "bold green" if pnl >= 0 else "bold red"
            pnl_txt = Text(f"{pnl_sign}${pnl:,.2f}", style=pnl_style)
            trade_count = tr_summary.get("total_today", 0)
            pnl_sub = Text(f"[{trade_count} trades]", style="dim")

            running = bot.get("running", False)
            stat_txt = Text("ACTIF" if running else "STOPPED",
                            style="bold cyan" if running else "bold red")
            session = mkt.get("session", "---")
            stat_sub = Text(f"[En session]" if session not in ("---", "Hors session", None) else "[Hors session]",
                            style="dim")

            kill_pct = acct.get("daily_pnl_pct", 0) or 0
            kill_val = abs(kill_pct) if kill_pct < 0 else 0.0
            is_danger = kill_val >= 3.0
            kill_txt = Text(f"{kill_val:.1f}%",
                            style="bold red on red" if is_danger else "bold bright_yellow")
            kill_sub = Text("DANGER" if is_danger else "Safe",
                            style="bold red on red" if is_danger else "bold bright_yellow")
        else:
            bal_txt = _disconnected_text()
            bal_sub = Text("")
            pnl_txt = _disconnected_text()
            pnl_sub = Text("")
            stat_txt = _disconnected_text()
            stat_sub = Text("")
            kill_txt = _disconnected_text()
            kill_sub = Text("")

        def _metric_panel(title: str, value: Text, sub: Text) -> Panel:
            g = Table.grid(padding=0)
            g.add_column(justify="center")
            g.add_row(value)
            g.add_row(sub)
            return Panel(g, title=f"[bold]{title}[/]", border_style="dim",
                         padding=(0, 1))

        mg_row = Table.grid(expand=True, padding=0)
        mg_row.add_column(ratio=1)
        mg_row.add_column(ratio=1)
        mg_row.add_column(ratio=1)
        mg_row.add_column(ratio=1)
        mg_row.add_row(
            _metric_panel("BALANCE", bal_txt, bal_sub),
            _metric_panel("P&L DU JOUR", pnl_txt, pnl_sub),
            _metric_panel("STATUT BOT", stat_txt, stat_sub),
            _metric_panel("KILL SWITCH", kill_txt, kill_sub),
        )
        parts.append(mg_row)

        # ── 3. REGIME MARCHE ────────────────────────────────────
        if api_ok:
            bias = mkt.get("daily_bias", "NEUTRAL")
            bias_style = {"BULLISH": "bold green", "BEARISH": "bold red"}.get(bias, "bold bright_yellow")

            atr = mkt.get("atr_m1")
            if atr is not None and atr > 10:
                atr = atr / 100
            atr_str = f"{atr:.2f}" if atr else "---"

            ema_slope = mkt.get("ema20_slope", 0) or 0
            slope_sign = "+" if ema_slope >= 0 else ""
            slope_style = "bold green" if ema_slope >= 0 else "bold red"

            session = mkt.get("session", "Hors session")
            sess_style = "bold cyan"
        else:
            bias = "---"
            bias_style = "bold red"
            atr_str = "---"
            ema_slope = 0
            slope_sign = ""
            slope_style = "dim"
            session = "---"
            sess_style = "dim"

        mkg = Table.grid(expand=True, padding=(0, 2))
        mkg.add_column(justify="center", ratio=1)
        mkg.add_column(justify="center", ratio=1)
        mkg.add_column(justify="center", ratio=1)
        mkg.add_column(justify="center", ratio=1)
        mkg.add_row(
            Text("DAILY BIAS", style="dim"),
            Text("ATR M1", style="dim"),
            Text("EMA20 SLOPE", style="dim"),
            Text("SESSION", style="dim"),
        )
        mkg.add_row(
            Text(bias, style=bias_style),
            Text(atr_str, style="bold white"),
            Text(f"{slope_sign}{ema_slope:.4f}" if api_ok else "---", style=slope_style),
            Text(session, style=sess_style),
        )
        parts.append(Panel(mkg, title="[bold]REGIME MARCHE[/]", border_style="dim", padding=(0, 0)))

        # ── 4. DERNIER TRADE ────────────────────────────────────
        if trades:
            last = trades[-1]
            d = last.get("direction", "---")
            d_style = "bold green" if d == "LONG" else "bold red"
            entry = f"${last['entry_price']:,.2f}" if last.get("entry_price") else "---"
            ep = last.get("exit_price")
            exit_str = f"${ep:,.2f}" if ep else "EN COURS"
            exit_style = "bold white" if ep else "bold cyan"
            lp = last.get("pnl")
            if lp is not None:
                lp_sign = "+" if lp >= 0 else ""
                lp_str = f"{lp_sign}${lp:,.2f}"
                lp_style = "bold green" if lp >= 0 else "bold red"
            else:
                lp_str = "---"
                lp_style = "dim"

            lg = Table.grid(expand=True, padding=(0, 2))
            lg.add_column(justify="center", ratio=1)
            lg.add_column(justify="center", ratio=1)
            lg.add_column(justify="center", ratio=1)
            lg.add_column(justify="center", ratio=1)
            lg.add_row(
                Text("DIRECTION", style="dim"),
                Text("ENTREE", style="dim"),
                Text("SORTIE", style="dim"),
                Text("P&L", style="dim"),
            )
            lg.add_row(
                Text(d, style=d_style),
                Text(entry, style="bold white"),
                Text(exit_str, style=exit_style),
                Text(lp_str, style=lp_style),
            )
            parts.append(Panel(lg, title="[bold]DERNIER TRADE[/]", border_style="dim", padding=(0, 0)))
        elif api_ok:
            parts.append(Panel(
                Align.center(Text("Aucun trade aujourd'hui", style="dim italic")),
                title="[bold]DERNIER TRADE[/]", border_style="dim", padding=(0, 0)))
        else:
            lg = Table.grid(expand=True)
            lg.add_column(justify="center")
            lg.add_row(_disconnected_text())
            parts.append(Panel(lg, title="[bold]DERNIER TRADE[/]", border_style="dim", padding=(0, 0)))

        # ── 5. TRADES DU JOUR (table) ──────────────────────────
        tt = Table(show_header=True, header_style="bold white", border_style="dim",
                   expand=True, padding=(0, 1))
        tt.add_column("HEURE", style="cyan", width=8)
        tt.add_column("DIR", width=7)
        tt.add_column("ENTREE", justify="right", width=12)
        tt.add_column("SORTIE", justify="right", width=12)
        tt.add_column("RAISON", width=12)
        tt.add_column("P&L", justify="right", width=12)

        if not trades:
            tt.add_row("--", "--", "--", "--", "--", "--")
        else:
            for t in trades:
                h = t["opened_at"][11:16] if t.get("opened_at") else "--"
                d = t.get("direction", "--")
                ds = "green" if d == "LONG" else "red"
                en = f"${t['entry_price']:,.2f}" if t.get("entry_price") else "--"
                ep = t.get("exit_price")
                if ep:
                    ex = f"${ep:,.2f}"
                else:
                    ex = "[bold cyan]EN COURS[/]"
                reason = t.get("reason", t.get("close_reason", "--"))
                p = t.get("pnl")
                if p is not None:
                    pst = "bold green" if p >= 0 else "bold red"
                    ps = f"[{pst}]{'+'if p>=0 else ''}${p:,.2f}[/]"
                elif ep is None:
                    ps = "[dim]---[/]"
                else:
                    ps = "[dim]---[/]"
                tt.add_row(h, f"[{ds}]{d}[/]", en, ex, str(reason), ps)

        parts.append(Panel(tt, title="[bold]TRADES DU JOUR[/]", border_style="dim", padding=(0, 0)))

        # ── 6. FOOTER ──────────────────────────────────────────
        if api_ok:
            bid = mkt.get("bid")
            bid_str = f"${bid:,.2f}" if bid else "---"
            lot = acct.get("lot_size", 0.02)
            spread = mkt.get("spread", 0)
            commission = acct.get("commission", 0.28)
            footer = Text()
            footer.append(f"XAUUSD {bid_str}", style="bold yellow")
            footer.append(f"  |  Lot {lot}  |  Spread {spread}  |  Commission ${commission}/trade", style="dim")
        else:
            footer = _disconnected_text()

        parts.append(Align.center(footer))

        # ── 7. PROMPT ──────────────────────────────────────────
        prompt = Text()
        prompt.append("inoxtrade@gold:~$ ", style="bold red")
        prompt.append("_", style="bold red blink")
        parts.append(prompt)

        return Group(*parts)

    console = Console(style=Style(bgcolor="#0d0d0d", color="white"))

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                status = _api_get_safe("/status")
                trades_data = _api_get_safe("/trades")
                dashboard = _build_dashboard(status, trades_data)
                live.update(dashboard)
                time.sleep(3)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Dashboard stopped.[/]")


# ---------------------------------------------------------------------------
# Interactive Menu
# ---------------------------------------------------------------------------

def cmd_menu():
    """Interactive terminal menu with arrow-key navigation."""
    import getpass
    import os

    import readchar
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    console = Console(style=Style(bgcolor="#0d0d0d", color="white"))

    # ── helpers ────────────────────────────────────────────────

    def _clear():
        console.clear()

    def _header():
        t = Text()
        t.append("INOXTRADE", style="bold bright_white")
        t.append(" — MENU", style="dim")
        console.print(Panel(Align.center(t), border_style="dim", padding=(0, 2)))

    def _select(title: str, options: list[str], *, allow_quit: bool = True) -> int | None:
        """Arrow-key menu. Returns index or None on quit/escape."""
        idx = 0
        while True:
            _clear()
            _header()
            console.print()
            if title:
                console.print(f"  [bold]{title}[/]\n")
            for i, opt in enumerate(options):
                if i == idx:
                    console.print(f"  [bold bright_white on grey23] > {opt} [/]")
                else:
                    console.print(f"    [dim]{opt}[/]")
            if allow_quit:
                console.print(f"\n  [dim]Fleches pour naviguer  |  Entree pour valider  |  q/Echap pour retour[/]")

            key = readchar.readkey()
            if key == readchar.key.UP:
                idx = (idx - 1) % len(options)
            elif key == readchar.key.DOWN:
                idx = (idx + 1) % len(options)
            elif key in (readchar.key.ENTER, "\r", "\n"):
                return idx
            elif key in ("q", "Q", readchar.key.ESC, "\x1b"):
                return None
            elif key.isdigit():
                n = int(key)
                if 1 <= n <= len(options):
                    return n - 1

    def _pause(msg: str = "Appuie sur une touche pour revenir..."):
        console.print(f"\n  [dim]{msg}[/]")
        readchar.readkey()

    def _not_available():
        console.print("  [bold yellow]Fonctionnalite disponible prochainement[/]")
        _pause()

    # ── [1] Strategie ──────────────────────────────────────────

    def _menu_strategy():
        status = _api_get_safe("/status")
        if status is None:
            _clear()
            _header()
            console.print("\n  [bold red]API DECONNECTEE[/]")
            _pause()
            return

        bot = status.get("bot", {})
        current = bot.get("strategy", "DTF")
        bot_running = bot.get("running", False)

        strategies = [
            ("DTF", "Daily Trend Follower"),
            ("ERM", "EMA Ribbon Momentum"),
            ("VRS", "VWAP Reversion Scalper"),
        ]

        labels = []
        for code, desc in strategies:
            tag = " (actuelle)" if code == current else ""
            labels.append(f"{code}  — {desc}{tag}")

        _clear()
        _header()
        console.print(f"\n  Strategie active : [bold cyan]{current}[/]\n")

        if bot_running:
            console.print("  [bold yellow]Bot actif — arrete-le d'abord.[/]")
            _pause()
            return

        choice = _select("Changer vers :", labels)
        if choice is None:
            return

        new_code = strategies[choice][0]
        if new_code == current:
            _clear()
            _header()
            console.print(f"\n  [dim]{new_code} est deja la strategie active.[/]")
            _pause()
            return

        result = _api_post_safe("/strategy", {"name": new_code})
        _clear()
        _header()
        if result and result.get("success"):
            console.print(f"\n  [bold green]Strategie changee → {new_code}. Redemarrage requis.[/]")
        elif result:
            console.print(f"\n  [bold red]Erreur : {result.get('message', 'inconnue')}[/]")
        else:
            _not_available()
            return
        _pause()

    # ── [2] Backtest ───────────────────────────────────────────

    def _menu_backtest():
        strats = ["DTF", "ERM", "VRS"]
        periods = [("7j", 7), ("30j", 30), ("90j", 90)]
        lots = [0.01, 0.02, 0.05]

        # Pick strategy
        choice_s = _select("Strategie :", strats)
        if choice_s is None:
            return
        strat = strats[choice_s]

        # Pick period
        choice_p = _select("Periode :", [p[0] for p in periods])
        if choice_p is None:
            return
        days = periods[choice_p][1]

        # Pick lot
        choice_l = _select("Lot size :", [str(l) for l in lots])
        if choice_l is None:
            return
        lot = lots[choice_l]

        # Confirm
        _clear()
        _header()
        console.print(f"\n  Strategie : [bold]{strat}[/]")
        console.print(f"  Periode   : [bold]{days}j[/]")
        console.print(f"  Lot       : [bold]{lot}[/]\n")

        # Launch
        result = _api_post_safe("/backtest", {"strategy": strat, "days": days, "lot": lot})

        if result is None:
            _not_available()
            return

        if result.get("async"):
            # Poll for result
            from rich.spinner import Spinner
            from rich.live import Live

            console.print("  [bold yellow]Backtest en cours... (peut prendre 2-5 min)[/]\n")
            with Live(Spinner("dots", text="  Calcul en cours..."), console=console, refresh_per_second=4):
                for _ in range(300):  # max 5 min
                    time.sleep(1)
                    poll = _api_get_safe("/backtest/status")
                    if poll and poll.get("done"):
                        result = _api_get_safe("/backtest/result")
                        break
                else:
                    console.print("  [bold red]Timeout — backtest trop long.[/]")
                    _pause()
                    return

        # Display result
        _clear()
        _header()

        r = result if result else {}
        tbl = Table(border_style="dim", padding=(0, 2), expand=False)
        tbl.add_column("", style="dim", width=12)
        tbl.add_column("", style="bold white", width=20)
        tbl.add_row("Trades", str(r.get("total_trades", "--")))
        wr = r.get("win_rate")
        tbl.add_row("Win Rate", f"{wr:.1f}%" if wr is not None else "--")
        pf = r.get("profit_factor")
        tbl.add_row("PF", f"{pf:.2f}" if pf is not None else "--")
        pnl = r.get("pnl")
        if pnl is not None:
            ps = "bold green" if pnl >= 0 else "bold red"
            sign = "+" if pnl >= 0 else ""
            tbl.add_row("P&L", f"[{ps}]{sign}${pnl:,.2f}[/]")
        else:
            tbl.add_row("P&L", "--")
        dd = r.get("max_drawdown")
        tbl.add_row("Max DD", f"${dd:,.2f}" if dd is not None else "--")

        console.print(Panel(
            tbl,
            title=f"[bold]BACKTEST {strat} — {days}j[/]",
            border_style="dim",
            padding=(0, 1),
        ))

        # Export option
        export_choice = _select("", ["Exporter ce rapport", "Retour"])
        if export_choice == 0:
            csv_path = Path.home() / "Desktop" / f"inoxtrade_backtest_{strat}_{days}j_{datetime.now().strftime('%Y%m%d')}.csv"
            try:
                import csv
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Metric", "Value"])
                    w.writerow(["Strategy", strat])
                    w.writerow(["Period", f"{days}j"])
                    w.writerow(["Lot", lot])
                    w.writerow(["Trades", r.get("total_trades", "")])
                    w.writerow(["Win Rate", f"{wr:.1f}%" if wr else ""])
                    w.writerow(["Profit Factor", f"{pf:.2f}" if pf else ""])
                    w.writerow(["P&L", f"{pnl:.2f}" if pnl else ""])
                    w.writerow(["Max Drawdown", f"{dd:.2f}" if dd else ""])
                _clear()
                _header()
                console.print(f"\n  [bold green]Exporte → {csv_path}[/]")
            except Exception as e:
                _clear()
                _header()
                console.print(f"\n  [bold red]Erreur export : {e}[/]")
            _pause()

    # ── [3] Compte MT5 ─────────────────────────────────────────

    def _menu_mt5():
        status = _api_get_safe("/status")

        _clear()
        _header()
        console.print()

        if status is None:
            console.print("  [bold red]API DECONNECTEE[/]")
            _pause()
            return

        acct = status.get("account", {})
        login_raw = str(acct.get("login", ""))
        login_masked = "*" * max(0, len(login_raw) - 2) + login_raw[-2:] if len(login_raw) > 2 else "********"

        tbl = Table(border_style="dim", padding=(0, 2), expand=False)
        tbl.add_column("", style="dim", width=14)
        tbl.add_column("", style="bold white", width=24)
        tbl.add_row("Login", login_masked)
        tbl.add_row("Server", str(acct.get("server", "--")))
        bal = acct.get("balance")
        tbl.add_row("Balance", f"${bal:,.2f}" if bal else "--")
        eq = acct.get("equity")
        tbl.add_row("Equity", f"${eq:,.2f}" if eq else "--")

        console.print(Panel(tbl, title="[bold]Compte actuel[/]", border_style="dim", padding=(0, 1)))

        choice = _select("", ["Changer de compte", "Retour"])
        if choice != 0:
            return

        # Credential input
        _clear()
        _header()
        console.print("\n  [bold]Nouveaux credentials MT5[/]\n")
        console.print("  [bold yellow]Ne jamais partager ces credentials.[/]\n")

        try:
            new_login = input("  Login : ").strip()
            new_pass = getpass.getpass("  Password : ")
        except (EOFError, KeyboardInterrupt):
            return

        servers = ["ICMarketsSC-Live", "ICMarketsSC-Demo", "Autre"]
        srv_idx = _select("Server :", servers)
        if srv_idx is None:
            return
        if srv_idx == 2:
            _clear()
            _header()
            try:
                new_server = input("\n  Nom du server : ").strip()
            except (EOFError, KeyboardInterrupt):
                return
        else:
            new_server = servers[srv_idx]

        if not new_login or not new_pass:
            _clear()
            _header()
            console.print("\n  [bold red]Login et password requis.[/]")
            _pause()
            return

        result = _api_post_safe("/mt5/credentials", {
            "login": new_login,
            "password": new_pass,
            "server": new_server,
        })

        _clear()
        _header()
        if result and result.get("success"):
            console.print("\n  [bold green]Credentials mis a jour. Redemarrage requis.[/]")
        elif result:
            console.print(f"\n  [bold red]Erreur : {result.get('message', 'inconnue')}[/]")
        else:
            _not_available()
            return
        _pause()

    # ── [4] Positions ──────────────────────────────────────────

    def _menu_positions():
        positions = _api_get_safe("/positions")

        _clear()
        _header()
        console.print()

        if positions is None:
            # Try /status fallback
            status = _api_get_safe("/status")
            if status is None:
                console.print("  [bold red]API DECONNECTEE[/]")
                _pause()
                return
            _not_available()
            return

        pos_list = positions.get("positions", [])
        count = len(pos_list)
        console.print(f"  Positions ouvertes : [bold]{count}[/]\n")

        if pos_list:
            tbl = Table(border_style="dim", expand=True, padding=(0, 1))
            tbl.add_column("Ticket", style="cyan", width=10)
            tbl.add_column("Dir", width=7)
            tbl.add_column("Entree", justify="right", width=10)
            tbl.add_column("Prix act", justify="right", width=10)
            tbl.add_column("P&L", justify="right", width=12)

            for p in pos_list:
                d = p.get("direction", "--")
                ds = "bold green" if d == "LONG" else "bold red"
                entry = f"${p['entry_price']:,.2f}" if p.get("entry_price") else "--"
                curr = f"${p['current_price']:,.2f}" if p.get("current_price") else "--"
                pnl = p.get("pnl")
                if pnl is not None:
                    ps = "bold green" if pnl >= 0 else "bold red"
                    sign = "+" if pnl >= 0 else ""
                    pnl_s = f"[{ps}]{sign}${pnl:,.2f}[/]"
                else:
                    pnl_s = "[dim]---[/]"
                tbl.add_row(str(p.get("ticket", "--")), f"[{ds}]{d}[/]", entry, curr, pnl_s)

            console.print(tbl)
            console.print()

            actions = ["Fermer toutes les positions", "Fermer une position", "Retour"]
        else:
            console.print("  [dim]Aucune position ouverte.[/]")
            _pause()
            return

        choice = _select("", actions)
        if choice is None or choice == 2:
            return

        if choice == 0:
            # Close all
            _clear()
            _header()
            try:
                confirm = input(f"\n  Fermer {count} position(s) ? (y/n) : ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if confirm != "y":
                return
            result = _api_post_safe("/positions/close_all")
            _clear()
            _header()
            if result and result.get("success"):
                closed = result.get("closed", count)
                console.print(f"\n  [bold green]{closed} position(s) fermee(s).[/]")
            elif result:
                console.print(f"\n  [bold red]Erreur : {result.get('message', 'inconnue')}[/]")
            else:
                _not_available()
                return
            _pause()

        elif choice == 1:
            # Close one
            _clear()
            _header()
            try:
                ticket = input("\n  Numero du ticket : ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not ticket:
                return
            result = _api_post_safe("/positions/close", {"ticket": int(ticket)})
            _clear()
            _header()
            if result and result.get("success"):
                console.print(f"\n  [bold green]Position {ticket} fermee.[/]")
            elif result:
                console.print(f"\n  [bold red]Erreur : {result.get('message', 'inconnue')}[/]")
            else:
                _not_available()
                return
            _pause()

    # ── [5] Exporter CSV ───────────────────────────────────────

    def _menu_export():
        periods = [
            ("Aujourd'hui", 1),
            ("7 derniers jours", 7),
            ("30 derniers jours", 30),
            ("Tout l'historique", 0),
        ]

        choice = _select("Exporter les trades :", [p[0] for p in periods])
        if choice is None:
            return

        days = periods[choice][1]
        params = {"format": "csv"}
        if days > 0:
            params["days"] = days

        _clear()
        _header()
        console.print("\n  [dim]Telechargement...[/]")

        data = _api_get_safe("/trades", params=params)

        if data is None:
            console.print("  [bold red]API DECONNECTEE[/]")
            _pause()
            return

        today = datetime.now().strftime("%Y%m%d")
        csv_path = Path.home() / "Desktop" / f"inoxtrade_trades_{today}.csv"

        try:
            if isinstance(data, str):
                csv_path.write_text(data)
            elif isinstance(data, dict):
                # API returned JSON — convert to CSV
                import csv as csv_mod
                trades = data.get("trades", [])
                if not trades:
                    console.print("  [dim]Aucun trade a exporter.[/]")
                    _pause()
                    return
                with open(csv_path, "w", newline="") as f:
                    w = csv_mod.DictWriter(f, fieldnames=trades[0].keys())
                    w.writeheader()
                    w.writerows(trades)
            _clear()
            _header()
            console.print(f"\n  [bold green]Exporte → {csv_path}[/]")
        except Exception as e:
            _clear()
            _header()
            console.print(f"\n  [bold red]Erreur : {e}[/]")
        _pause()

    # ── Main loop ──────────────────────────────────────────────

    MENU_ITEMS = [
        "Strategie",
        "Backtest",
        "Compte MT5",
        "Positions",
        "Exporter CSV",
        "Quitter",
    ]

    HANDLERS = [
        _menu_strategy,
        _menu_backtest,
        _menu_mt5,
        _menu_positions,
        _menu_export,
    ]

    while True:
        choice = _select("", [f"[{i+1}] {m}" if i < 5 else f"[q] {m}" for i, m in enumerate(MENU_ITEMS)])
        if choice is None or choice == 5:
            _clear()
            console.print("[dim]Au revoir.[/]")
            break
        HANDLERS[choice]()


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
COMMANDS = {
    "setup": cmd_setup,
    "connect": cmd_connect,
    "menu": cmd_menu,
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
        c.print("  [bold]menu[/]      Interactive menu")
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
