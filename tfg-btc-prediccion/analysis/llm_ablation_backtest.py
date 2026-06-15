"""
analysis/llm_ablation_backtest.py
─────────────────────────────────────────────────────────────────────────────
Ablation study: backtest del Full System CON vs SIN validación por agente LLM.

Cuando el confluence score supera el umbral (señal técnica), llamamos a OpenAI
GPT-4o-mini para que valide o rechace el setup en base a contexto cualitativo.

Hipótesis: el LLM filtra entradas malas que pasan el score numérico pero fallan
criterios cualitativos (e.g., contra-tendencia ICT, sin OB cercano, structure
mixed). Si la hipótesis es correcta, el sistema CON LLM debería:
  - Hacer MENOS trades
  - Mayor win rate
  - Mayor expectancy
  - Mayor Sharpe

Output:
    results/llm_ablation.csv             — comparativa
    results/llm_ablation_rejections.csv  — análisis de rechazos del LLM

Uso:
    python analysis/llm_ablation_backtest.py --days 365 --max-llm-calls 50
"""
from __future__ import annotations
import argparse, json, logging, os, sys, hashlib
from datetime import datetime, timezone
from importlib import util as _ilu
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "config" / ".env")

_spec = _ilu.spec_from_file_location("bt", ROOT / "analysis" / "backtest_confluence.py")
bt = _ilu.module_from_spec(_spec); sys.modules["bt"] = bt; _spec.loader.exec_module(bt)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm_ablation")

# ── LLM CACHE ─────────────────────────────────────────────────────────────────
CACHE_FILE = ROOT / "results" / "llm_cache.json"
_llm_cache = {}
if CACHE_FILE.exists():
    try:
        _llm_cache = json.loads(CACHE_FILE.read_text())
    except Exception:
        _llm_cache = {}

def _save_cache():
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_llm_cache))

# ── LLM CALL ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista experto en trading ICT que filtra setups por calidad.
Tu tarea: dado un setup que ya pasó el filtro técnico cuantitativo (score >= 0.30),
decides si CUALITATIVAMENTE vale la pena.

Aprueba (TRADE) cuando el setup tiene al menos UNO de estos refuerzos:
- Sweep o BOS/CHoCH reciente en dirección del trade
- Zona discount con intent long, O premium con intent short
- Cerca de liquidity pool diario/semanal en la dirección
- Confluence >= 0.45 (señal fuerte aunque sin contexto adicional)

Rechaza (NO_TRADE) solo cuando:
- Premium zone + intent LONG, O discount zone + intent SHORT (contra zona)
- Trend ICT en CONTRA de la dirección del trade
- Sin ningún refuerzo cualitativo Y confluence < 0.40
- BBW > 8% (volatilidad extrema, alto riesgo de whipsaw)

Sé balanceado: aprueba setups decentes, rechaza solo los claramente débiles.

Responde SOLO con JSON exacto:
{"decision":"TRADE"|"NO_TRADE","reason":"<1 frase breve>"}"""


def llm_decide(score: float, direction: str, ict_score: float, mtf_score: float,
               tech_score: float, near_pool: bool, in_premium: bool, in_discount: bool,
               trend: str, bbw_pct: float, has_recent_sweep: bool,
               has_recent_bos: bool, has_recent_choch: bool) -> dict:
    """Llama OpenAI con contexto del setup. Cachea por hash del input."""
    key_dict = {
        "s": round(score, 2), "d": direction,
        "ict": round(ict_score, 2), "mtf": round(mtf_score, 2), "tech": round(tech_score, 2),
        "near_pool": near_pool, "premium": in_premium, "discount": in_discount,
        "trend": trend, "bbw": round(bbw_pct, 1),
        "sweep": has_recent_sweep, "bos": has_recent_bos, "choch": has_recent_choch,
    }
    cache_key = hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    user_prompt = f"""Setup:
- Dirección propuesta: {direction.upper()}
- Confluence score: {score:+.2f}
- ICT score: {ict_score:+.2f} | MTF: {mtf_score:+.2f} | Tech: {tech_score:+.2f}
- Trend ICT: {trend}
- Cerca de liquidity pool: {near_pool}
- En premium zone: {in_premium} | En discount zone: {in_discount}
- Bollinger Band Width: {bbw_pct:.1f}%
- Recent sweep: {has_recent_sweep} | Recent BOS: {has_recent_bos} | Recent CHoCH: {has_recent_choch}

Decide: ¿TRADE o NO_TRADE?"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        text_resp = resp.choices[0].message.content
        decision = json.loads(text_resp)
        decision["_cost_tokens"] = resp.usage.total_tokens
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        decision = {"decision": "TRADE", "reason": "LLM error - default permissive", "_cost_tokens": 0}

    _llm_cache[cache_key] = decision
    _save_cache()
    return decision


def strategy_full_system_with_llm(df, fg_df, daily_feat, max_calls=100,
                                   threshold=bt.THRESHOLD, weights=None):
    """Full system + validación LLM cuando score cruza umbral."""
    llm_calls_used = 0
    rejections = []   # (timestamp, score, direction, reason)
    approvals  = []

    def signal_fn(i):
        nonlocal llm_calls_used
        if i < 50: return None
        ts = df.index[i]
        row = df.iloc[i]
        price = float(row["close"])

        s, scores = bt.confluence_score(ts, df, i, fg_df, ml_score=0.0, weights=weights)
        if abs(s) < threshold:
            return None
        direction = "long" if s > 0 else "short"

        # Stop calling LLM si pasamos el cap
        if llm_calls_used >= max_calls:
            return direction

        # Recoger contexto para LLM
        bb_u, bb_l = row.get("bb_upper"), row.get("bb_lower")
        bbw_pct = (float(bb_u) - float(bb_l)) / price * 100 if pd.notna(bb_u) and pd.notna(bb_l) and price else 0
        df_slice = df.iloc[max(0, i-10):i+1]
        has_sweep = bool(df_slice["liq_sweep_bull"].any() or df_slice["liq_sweep_bear"].any()) if "liq_sweep_bull" in df.columns else False
        has_bos = bool(df_slice["bos_bull"].any() or df_slice["bos_bear"].any()) if "bos_bull" in df.columns else False
        has_choch = bool(df_slice["choch_bull"].any() or df_slice["choch_bear"].any()) if "choch_bull" in df.columns else False

        # Llamar LLM
        decision = llm_decide(
            score=s, direction=direction,
            ict_score=scores.get("ict", 0), mtf_score=scores.get("mtf", 0),
            tech_score=scores.get("technical", 0),
            near_pool=bool(row.get("near_liquidity_pool", False)),
            in_premium=bool(row.get("is_premium", False)),
            in_discount=bool(row.get("is_discount", False)),
            trend=str(row.get("trend", "side")),
            bbw_pct=bbw_pct,
            has_recent_sweep=has_sweep, has_recent_bos=has_bos, has_recent_choch=has_choch,
        )
        llm_calls_used += 1

        rec = {"timestamp": ts, "score": round(s, 3), "direction": direction,
               "decision": decision.get("decision"), "reason": decision.get("reason", "")[:120]}
        if decision.get("decision") == "TRADE":
            approvals.append(rec)
            return direction
        else:
            rejections.append(rec)
            return None

    trades, eq = bt.simulate_strategy(df, signal_fn, money_mgmt=True)
    log.info(f"LLM calls: {llm_calls_used} | approved: {len(approvals)} | rejected: {len(rejections)}")
    return trades, eq, rejections, approvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--max-llm-calls", type=int, default=100)
    args = ap.parse_args()

    log.info("═" * 78)
    log.info(f"LLM ABLATION — {args.days} días {args.tf} | max LLM calls: {args.max_llm_calls}")
    log.info("═" * 78)

    df = bt.load_ohlcv(args.tf, args.days)
    df = bt.compute_indicators(df)
    df = bt.detect_obs_fvgs(df)
    if bt.ICT_ADVANCED_AVAILABLE:
        df = bt.detect_liquidity_sweeps(df)
        df = bt.detect_bos_choch(df)
        df = bt.compute_premium_discount(df)
        df = bt.detect_daily_weekly_levels(df)
    df = df.dropna(subset=["rsi_14", "macd", "ema_50", "atr_14"])
    fg = bt.load_fear_greed_daily()
    daily = bt.load_daily_features()
    period_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    log.info(f"Datos: {len(df)} bars 4h | {df.index[0].date()} → {df.index[-1].date()}")

    results, equities = {}, {}

    # 1. Sin LLM (baseline)
    log.info("\n→ Full System (sin LLM)")
    trades, eq = bt.strategy_full_system(df, fg, daily, threshold=bt.THRESHOLD)
    m = bt.compute_metrics(trades, eq, bt.INITIAL_CAPITAL, period_days)
    results["Sin LLM"] = m
    equities["Sin LLM"] = eq
    log.info(f"  ret={m['total_return_pct']:+.2f}% | sharpe={m['sharpe']:.2f} | trades={m['n_trades']} | win={m['win_rate_pct']}%")

    # 2. Con LLM
    log.info(f"\n→ Full System + LLM (max {args.max_llm_calls} calls)")
    trades, eq, rej, app = strategy_full_system_with_llm(df, fg, daily, max_calls=args.max_llm_calls)
    m = bt.compute_metrics(trades, eq, bt.INITIAL_CAPITAL, period_days)
    results["Con LLM"] = m
    equities["Con LLM"] = eq
    log.info(f"  ret={m['total_return_pct']:+.2f}% | sharpe={m['sharpe']:.2f} | trades={m['n_trades']} | win={m['win_rate_pct']}%")

    # Output
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    df_cmp = pd.DataFrame(results).T
    df_cmp.to_csv(out / "llm_ablation.csv")

    if rej:
        pd.DataFrame(rej).to_csv(out / "llm_ablation_rejections.csv", index=False)
    if app:
        pd.DataFrame(app).to_csv(out / "llm_ablation_approvals.csv", index=False)

    print()
    print("=" * 130)
    print("ABLATION: Full System SIN LLM vs CON LLM")
    print("=" * 130)
    cols = ["total_return_pct", "sharpe", "sortino", "max_dd_pct", "profit_factor",
            "win_rate_pct", "n_trades", "expectancy_usd", "p_value_vs_0"]
    print(df_cmp[cols].to_string())
    print()

    # Análisis incremental
    print("=" * 130)
    print("VALOR INCREMENTAL DEL LLM")
    print("=" * 130)
    sin = results["Sin LLM"]; con = results["Con LLM"]
    print(f"  Δ Return:      {con['total_return_pct'] - sin['total_return_pct']:+.2f}%")
    print(f"  Δ Sharpe:      {con['sharpe'] - sin['sharpe']:+.2f}")
    print(f"  Δ Max DD:      {con['max_dd_pct'] - sin['max_dd_pct']:+.2f}%")
    print(f"  Δ Win rate:    {con['win_rate_pct'] - sin['win_rate_pct']:+.1f}%")
    print(f"  Δ # trades:    {int(con['n_trades']) - int(sin['n_trades'])} (LLM filtró {int(sin['n_trades']) - int(con['n_trades'])} setups)")
    print(f"  Δ Profit Factor: {con['profit_factor'] - sin['profit_factor']:+.2f}" if isinstance(con['profit_factor'], (int,float)) and isinstance(sin['profit_factor'], (int,float)) else "")

    if rej:
        print()
        print(f"📋 LLM rechazó {len(rej)} setups. Ejemplos:")
        for r in rej[:5]:
            print(f"  · {r['timestamp']}  score={r['score']:+.2f} {r['direction']:5s}  → {r['reason']}")
    print()


if __name__ == "__main__":
    main()
