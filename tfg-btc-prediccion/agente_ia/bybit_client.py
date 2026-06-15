"""
bybit_client.py — Wrapper del API Bybit Demo Trading.
Soporta pybit (si disponible) o requests + HMAC-SHA256 directo.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config" / ".env")

log = logging.getLogger(__name__)

DEMO_BASE   = os.getenv("BYBIT_DEMO_BASE_URL", "https://api-demo.bybit.com")
API_KEY     = os.getenv("BYBIT_DEMO_API_KEY", "")
API_SECRET  = os.getenv("BYBIT_DEMO_API_SECRET", "")

SYMBOL      = "BTCUSDT"
CATEGORY    = "linear"
MIN_QTY_BTC = 0.001   # mínimo en BTCUSDT linear


# ── Firma HMAC-SHA256 ─────────────────────────────────────────────────────────

def _sign(params: dict, recv_window: int = 5000) -> dict:
    """Añade timestamp, api_key, recv_window y firma al dict de params."""
    ts = str(int(time.time() * 1000))
    params_copy = dict(params)
    params_copy.update({"api_key": API_KEY, "timestamp": ts, "recv_window": recv_window})
    sorted_str = "&".join(f"{k}={v}" for k, v in sorted(params_copy.items()))
    sig = hmac.new(API_SECRET.encode(), sorted_str.encode(), hashlib.sha256).hexdigest()
    params_copy["sign"] = sig
    return params_copy


def _headers() -> dict:
    ts = str(int(time.time() * 1000))
    recv_window = "5000"
    pre_sign = ts + API_KEY + recv_window
    sig = hmac.new(API_SECRET.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
    return {
        "X-BAPI-API-KEY":     API_KEY,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        sig,
        "Content-Type":       "application/json",
    }


def _get(path: str, params: dict = None) -> dict:
    url = DEMO_BASE + path
    if params is None:
        params = {}
    ts = str(int(time.time() * 1000))
    recv_window = "5000"
    # Build query string for signing
    qstr = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    pre_sign = ts + API_KEY + recv_window + qstr
    sig = hmac.new(API_SECRET.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     API_KEY,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        sig,
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    import json as _json
    url = DEMO_BASE + path
    ts = str(int(time.time() * 1000))
    recv_window = "5000"
    body_str = _json.dumps(body, separators=(",", ":"))
    pre_sign = ts + API_KEY + recv_window + body_str
    sig = hmac.new(API_SECRET.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     API_KEY,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        sig,
        "Content-Type":       "application/json",
    }
    r = requests.post(url, data=body_str, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Cliente principal ─────────────────────────────────────────────────────────

class BybitDemoClient:
    """Wrapper de Bybit Demo API v5 con firma HMAC-SHA256."""

    def __init__(self):
        self._api_key    = API_KEY
        self._api_secret = API_SECRET
        self._base       = DEMO_BASE

        if not self._api_key or not self._api_secret:
            log.warning("Bybit Demo: BYBIT_DEMO_API_KEY o SECRET no configurados en .env")

        # Intentar pybit primero (más robusto)
        self._session = None
        try:
            from pybit.unified_trading import HTTP
            self._session = HTTP(
                testnet=False,
                demo=True,
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
            log.info("Bybit cliente: usando pybit SDK")
        except Exception as e:
            log.info("pybit no disponible (%s) — usando requests HMAC directo", e)

    # ── Precio ────────────────────────────────────────────────────────────────

    def get_price(self) -> float:
        """Precio de mercado actual BTCUSDT."""
        try:
            if self._session:
                r = self._session.get_tickers(category=CATEGORY, symbol=SYMBOL)
                return float(r["result"]["list"][0]["lastPrice"])
        except Exception:
            pass
        # Fallback: pública (sin auth)
        url = self._base + "/v5/market/tickers"
        r = requests.get(url, params={"category": CATEGORY, "symbol": SYMBOL}, timeout=10)
        r.raise_for_status()
        return float(r.json()["result"]["list"][0]["lastPrice"])

    # ── Balance ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Equity total en USDT (cuenta UNIFIED)."""
        self._require_auth()
        try:
            if self._session:
                r = self._session.get_wallet_balance(accountType="UNIFIED")
                return float(r["result"]["list"][0]["totalEquity"])
        except Exception:
            pass
        r = _get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        _check(r)
        return float(r["result"]["list"][0]["totalEquity"])

    # ── Posiciones ────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list:
        """Lista de posiciones abiertas en BTCUSDT."""
        self._require_auth()
        try:
            if self._session:
                r = self._session.get_positions(category=CATEGORY, symbol=SYMBOL)
                return [p for p in r["result"]["list"] if float(p.get("size", 0)) > 0]
        except Exception:
            pass
        r = _get("/v5/position/list", {"category": CATEGORY, "symbol": SYMBOL})
        _check(r)
        return [p for p in r["result"]["list"] if float(p.get("size", 0)) > 0]

    # ── Leverage ──────────────────────────────────────────────────────────────

    def set_leverage(self, leverage: float) -> None:
        lev = str(int(leverage))
        try:
            if self._session:
                self._session.set_leverage(
                    category=CATEGORY, symbol=SYMBOL,
                    buyLeverage=lev, sellLeverage=lev,
                )
                return
        except Exception:
            pass
        _post("/v5/position/set-leverage", {
            "category": CATEGORY, "symbol": SYMBOL,
            "buyLeverage": lev, "sellLeverage": lev,
        })

    # ── Abrir posición ────────────────────────────────────────────────────────

    def open_position(
        self,
        direction: str,          # 'long' | 'short'
        size_usd: float,
        leverage: float = 3.0,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> dict:
        """
        Abre posición market en BTCUSDT con SL/TP nativos.
        Devuelve {order_id, status, side, size_btc, msg}.
        """
        self._require_auth()

        price = self.get_price()
        size_btc = round(size_usd / price / leverage, 3)
        if size_btc < MIN_QTY_BTC:
            size_btc = MIN_QTY_BTC
            log.warning("Tamaño ajustado al mínimo: %.3f BTC", size_btc)

        side = "Buy" if direction == "long" else "Sell"

        try:
            self.set_leverage(leverage)
        except Exception as e:
            log.warning("set_leverage falló (puede ya estar configurado): %s", e)

        body = {
            "category":   CATEGORY,
            "symbol":     SYMBOL,
            "side":       side,
            "orderType":  "Market",
            "qty":        str(size_btc),
            "timeInForce": "IOC",
            "reduceOnly": False,
        }
        if stop_loss:
            body["stopLoss"]  = str(round(stop_loss, 2))
        if take_profit:
            body["takeProfit"] = str(round(take_profit, 2))

        try:
            if self._session:
                r = self._session.place_order(**body)
            else:
                r = _post("/v5/order/create", body)
        except Exception as e:
            log.error("open_position error: %s", e)
            return {"error": str(e), "status": "rejected"}

        _check(r)
        order_id = r.get("result", {}).get("orderId", "")
        log.info("Posición abierta: %s %s %.3f BTC | order=%s", side, SYMBOL, size_btc, order_id)
        return {
            "order_id":    order_id,
            "side":        side,
            "size_btc":    size_btc,
            "entry_price": price,
            "status":      "open",
        }

    # ── Mover Stop Loss nativo (breakeven / trailing) ──────────────────────────

    def update_stop_loss(self, stop_loss: float) -> dict:
        """Modifica el SL nativo de la posición abierta en Bybit (set-trading-stop).
        Persiste en el exchange, así protege aunque el bot se apague."""
        self._require_auth()
        sl_str = str(round(float(stop_loss), 2))
        body = {
            "category":    CATEGORY,
            "symbol":      SYMBOL,
            "stopLoss":    sl_str,
            "tpslMode":    "Full",
            "positionIdx": 0,
        }
        try:
            if self._session:
                r = self._session.set_trading_stop(**body)
            else:
                r = _post("/v5/position/trading-stop", body)
        except Exception as e:
            log.error("update_stop_loss error: %s", e)
            return {"error": str(e)}
        _check(r)
        log.info("SL movido a %s en Bybit", sl_str)
        return {"status": "ok", "stop_loss": float(sl_str)}

    # ── Cerrar posición ───────────────────────────────────────────────────────

    def close_position(self, direction: str) -> dict:
        """Cierra posición con orden de mercado reduceOnly."""
        self._require_auth()

        positions = self.get_open_positions()
        if not positions:
            return {"status": "no_position"}

        pos = positions[0]
        size_btc = abs(float(pos.get("size", 0)))
        if size_btc == 0:
            return {"status": "no_position"}

        close_side = "Sell" if direction == "long" else "Buy"
        body = {
            "category":    CATEGORY,
            "symbol":      SYMBOL,
            "side":        close_side,
            "orderType":   "Market",
            "qty":         str(size_btc),
            "timeInForce": "IOC",
            "reduceOnly":  True,
        }
        try:
            if self._session:
                r = self._session.place_order(**body)
            else:
                r = _post("/v5/order/create", body)
        except Exception as e:
            log.error("close_position error: %s", e)
            return {"error": str(e)}

        _check(r)
        price = self.get_price()
        log.info("Posición cerrada: %s %s %.3f BTC @ %.2f", close_side, SYMBOL, size_btc, price)
        return {"status": "closed", "exit_price": price, "size_btc": size_btc}

    def close_all(self) -> list:
        """Cierra todas las posiciones abiertas."""
        results = []
        for pos in self.get_open_positions():
            side = pos.get("side", "Buy")
            direction = "long" if side == "Buy" else "short"
            results.append(self.close_position(direction))
        return results

    # ── PnL no realizado ──────────────────────────────────────────────────────

    def get_unrealized_pnl(self) -> float:
        """PnL no realizado total de posiciones abiertas."""
        positions = self.get_open_positions()
        return sum(float(p.get("unrealisedPnl", 0)) for p in positions)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_auth(self):
        if not self._api_key or not self._api_secret:
            raise RuntimeError(
                "Bybit Demo: configura BYBIT_DEMO_API_KEY y BYBIT_DEMO_API_SECRET en config/.env"
            )

    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_secret)


def _check(r: dict):
    ret_code = r.get("retCode", -1)
    if ret_code != 0:
        raise RuntimeError(f"Bybit error {ret_code}: {r.get('retMsg','?')}")


# ── Test directo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    client = BybitDemoClient()
    print("Precio BTC:", client.get_price())
    if client.is_configured():
        print("Balance:", client.get_balance(), "USDT")
        print("Posiciones abiertas:", json.dumps(client.get_open_positions(), indent=2))
    else:
        print("⚠️  Añade BYBIT_DEMO_API_KEY y BYBIT_DEMO_API_SECRET en config/.env para usar el bot")
