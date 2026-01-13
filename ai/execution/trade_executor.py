# ai/execution/trade_executor.py
from __future__ import annotations
import time
from typing import Dict, Optional, Tuple
import pandas as pd


from app_config import CFG  # centralized app_config


class TradeExecutor:
    """
    Alpaca paper-trading executor.
    Converts target allocations into buy/sell orders, and
    sends Telegram alerts for each trade with full details.
    """

    def __init__(self):
        # Alpaca credentials via app_config.py
        self.api_key = CFG.alpaca.key
        self.secret = CFG.alpaca.secret
        self.base_url = CFG.alpaca.base_url
        self.api = tradeapi.REST(self.api_key, self.secret, self.base_url, api_version="v2")

        # Telegram
        self._tg_token = CFG.telegram.token
        self._tg_chat = CFG.telegram.chat_id
        self._notify_enabled = bool(self._tg_token and self._tg_chat)

    # ----------------------------------------------------------------
    # Telegram helpers
    # ----------------------------------------------------------------
    def _send_telegram(self, text: str, html: bool = True):
        """Send a Telegram message if configured."""
        if not self._notify_enabled:
            return
        import requests
        try:
            url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
            payload = {"chat_id": self._tg_chat, "text": text, "parse_mode": "HTML" if html else "Markdown"}
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"⚠️ Telegram send failed: {e}")

    # ----------------------------------------------------------------
    # Positions / cash
    # ----------------------------------------------------------------
    def get_positions(self) -> Dict[str, float]:
        """Return current position sizes (symbol -> quantity)."""
        positions = {}
        for p in self.api.list_positions():
            positions[p.symbol] = float(p.qty)
        return positions

    def get_cash(self) -> float:
        """Fetch available cash balance."""
        account = self.api.get_account()
        return float(account.cash)

    # ----------------------------------------------------------------
    # Option symbol parsing (OCC format)
    #   Example: AAPL240118C00190000  ->  Exp: 2024-01-18, Call, Strike 190.00
    #   If not option or unparseable, returns None fields gracefully.
    # ----------------------------------------------------------------
    @staticmethod
    def _parse_occ_symbol(sym: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        try:
            # Very defensive: OCC formatted options are typically 15 chars after root
            # Pattern: ROOT(1-6) + YYMMDD(6) + C/P(1) + STRIKE(8) (e.g., 00190000 => 190.0000)
            # We’ll search from the end to be safer when ROOT varies.
            # Find last 15 characters:
            tail = sym[-15:]
            yymmdd = tail[:6]
            cp = tail[6]
            strike_raw = tail[7:]
            # parse
            year = int("20" + yymmdd[:2])
            month = int(yymmdd[2:4])
            day = int(yymmdd[4:6])
            exp = f"{year:04d}-{month:02d}-{day:02d}"
            right = "CALL" if cp.upper() == "C" else ("PUT" if cp.upper() == "P" else None)
            strike = float(int(strike_raw) / 1000.0)  # 8 digits with 3 decimals
            return exp, right, strike
        except Exception:
            return None, None, None

    # ----------------------------------------------------------------
    # Alert formatter for orders
    # ----------------------------------------------------------------
    def _format_order_alert(
        self,
        order: tradeapi.entity.Order,
        side: str,
        qty: float,
        est_price: float,
    ) -> str:
        """
        Produce a rich Telegram message for a trade.
        Includes option details if the symbol looks like an OCC option.
        """
        sym = order.symbol or ""
        exp, right, strike = self._parse_occ_symbol(sym)
        asset_class = getattr(order, "asset_class", "us_equity")

        submitted_at = getattr(order, "submitted_at", None)
        filled_at = getattr(order, "filled_at", None)
        status = getattr(order, "status", "new") or "new"
        tif = getattr(order, "time_in_force", "day") or "day"
        otype = getattr(order, "type", "market") or "market"
        order_id = getattr(order, "id", "") or ""
        client_oid = getattr(order, "client_order_id", "") or ""

        # Prefer real filled price if available
        filled_avg_price = None
        try:
            fap = getattr(order, "filled_avg_price", None)
            filled_avg_price = float(fap) if fap else None
        except Exception:
            filled_avg_price = None
        px = filled_avg_price or est_price

        # timestamps to human readable UTC
        def _fmt_ts(ts):
            if not ts:
                return "-"
            try:
                dt = pd.to_datetime(ts, utc=True)
                return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                return str(ts)

        # Build message lines
        lines = []
        lines.append("📣 <b>TRADE EXECUTED</b>")
        lines.append(f"• <b>Side:</b> {side.upper()}")
        lines.append(f"• <b>Symbol:</b> {sym}")
        lines.append(f"• <b>Qty:</b> {qty}")
        lines.append(f"• <b>Price:</b> ${px:,.2f}")
        lines.append(f"• <b>Type / TIF:</b> {otype.upper()} / {tif.upper()}")
        lines.append(f"• <b>Status:</b> {status}")
        if asset_class and asset_class.lower() != "us_equity":
            lines.append(f"• <b>Asset Class:</b> {asset_class}")
        # Option extras (if we can parse them)
        if exp or right or strike:
            if exp:
                lines.append(f"• <b>Expiration:</b> {exp}")
            if right:
                lines.append(f"• <b>Right:</b> {right}")
            if strike:
                lines.append(f"• <b>Strike:</b> {strike:,.2f}")
        # IDs + times
        if order_id:
            lines.append(f"• <b>Order ID:</b> <code>{order_id}</code>")
        if client_oid:
            lines.append(f"• <b>Client OID:</b> <code>{client_oid}</code>")
        lines.append(f"• <b>Submitted:</b> {_fmt_ts(submitted_at)}")
        lines.append(f"• <b>Filled:</b> {_fmt_ts(filled_at)}")

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Main execution (target allocations)
    # ----------------------------------------------------------------
    def place_orders(self, target_allocs: Dict[str, float]):
        """
        Target-based allocation rebalancing.
        Executes paper orders via Alpaca and sends Telegram alerts.
        """
        if not target_allocs:
            print("⚠️ No allocations to execute.")
            return

        # Portfolio snapshot
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        current_positions = self.get_positions()

        # Latest quotes (fallback to 0 if not available)
        quotes = {}
        for sym in target_allocs.keys():
            try:
                # Using last trade price (IEX) as a simple proxy
                quotes[sym] = float(self.api.get_latest_trade(sym).price)
            except Exception:
                quotes[sym] = 0.0

        # Rebalance loop
        for sym, weight in target_allocs.items():
            price = quotes.get(sym, 0)
            if price <= 0:
                print(f"⚠️ Skipping {sym}: no valid price quote.")
                continue

            target_value = portfolio_value * weight
            target_qty = int(target_value / price)
            current_qty = int(current_positions.get(sym, 0))
            diff = target_qty - current_qty

            if abs(diff) < 1:
                continue  # trivial

            side = "buy" if diff > 0 else "sell"
            qty = abs(diff)

            try:
                order = self.api.submit_order(
                    symbol=sym,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="day",
                )
                # Console
                print(f"✅ {side.upper()} {qty} {sym} @ est ${price:.2f}")
                # Telegram (rich)
                try:
                    alert = self._format_order_alert(order, side=side, qty=qty, est_price=price)
                    self._send_telegram(alert)
                except Exception as ne:
                    print(f"⚠️ Alert format/send failed: {ne}")
                time.sleep(0.25)
            except Exception as e:
                print(f"⚠️ Order for {sym} failed: {e}")
