"""Streamlit dashboard for Lyntrix swarm and paper-trading state."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
import json
from typing import Any

import pandas as pd
import streamlit as st

from shared.database import Database


def run_async(coro: Any) -> Any:
    """Run async coroutine from Streamlit's synchronous context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def normalize_payload(value: Any) -> dict[str, Any]:
    """Normalize JSON payload value to dictionary."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def build_agent_rows(
    reports: list[dict[str, Any]],
    asset_filter: str,
) -> list[dict[str, Any]]:
    """Build table rows from latest per-agent reports."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        payload = normalize_payload(report.get("payload"))
        impacted = payload.get("impacted_sectors", [])
        summary = str(payload.get("summary", payload.get("reason", "-")))
        confidence = payload.get("confidence", "-")
        asset = str(payload.get("asset", "-"))
        if asset_filter != "ALL":
            if asset_filter == "COPPER" and asset != "HG=F":
                continue
            if asset_filter == "GOLD" and asset != "GC=F":
                continue
            if asset_filter == "ENERGY":
                energy_names = {"NLR", "XLU", "ENERGY", "ETF"}
                token = summary.upper() + " " + asset.upper()
                if not any(name in token for name in energy_names):
                    continue

        rows.append(
            {
                "agent": report.get("agent", "-"),
                "timestamp": report.get("ts"),
                "confidence": confidence,
                "summary": summary,
                "signal": payload.get("signal", payload.get("action", "-")),
                "asset": asset,
                "impacted_sectors": impacted,
            }
        )
    return rows


def render_dashboard() -> None:
    """Render one full dashboard refresh frame."""
    st.set_page_config(
        page_title="Lyntrix Invest Bot Dashboard",
        layout="wide",
    )
    st.title("Lyntrix Invest Bot - Swarm Console")

    with st.sidebar:
        st.header("Filtros")
        asset_filter = st.selectbox(
            "Activo",
            options=["ALL", "COPPER", "GOLD", "ENERGY"],
            index=0,
        )
        default_start = date.today() - timedelta(days=30)
        start_date = st.date_input("Desde", value=default_start)
        end_date = st.date_input(
            "Hasta",
            value=date.today() + timedelta(days=1),
        )
        trades_limit = st.slider("Max trades", min_value=20, max_value=500, value=200)

    db = Database()
    symbol_map = {
        "ALL": None,
        "COPPER": "HG=F",
        "GOLD": "GC=F",
        "ENERGY": "NLR",
    }
    try:
        snapshot = run_async(db.fetch_latest_portfolio_snapshot())
        history = run_async(db.fetch_portfolio_history(start_date, end_date))
        reports = run_async(db.fetch_latest_agent_reports())
        decisions = run_async(
            db.fetch_final_decisions(start_date, end_date, limit=50),
        )
        positions = run_async(db.fetch_open_positions())
        trades = run_async(
            db.fetch_executed_trades(
                start_date=start_date,
                end_date=end_date,
                symbol=symbol_map[asset_filter],
                limit=trades_limit,
            )
        )
        macro_regime = run_async(db.fetch_latest_macro_regime()) or "UNKNOWN"
        black_swan_risk = run_async(db.fetch_latest_black_swan_risk()) or "LOW"
    finally:
        run_async(db.close())

    if black_swan_risk.upper() == "CRITICAL":
        st.error(
            "BLACK SWAN ALERT: riesgo CRITICAL detectado. "
            "Modo defensivo activo (ABORT/STAY_OUT).",
            icon="🚨",
        )

    if snapshot is None:
        current_balance = 0.0
        pnl_pct = 0.0
    else:
        current_balance = float(snapshot["total_equity"])
        baseline = current_balance
        if history:
            baseline = float(history[0]["total_equity"])
        pnl_pct = ((current_balance / max(baseline, 1e-8)) - 1.0) * 100.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Balance Actual", f"${current_balance:,.2f}")
    m2.metric("PnL Total (%)", f"{pnl_pct:+.2f}%")
    m3.metric("Regimen Macro", macro_regime)

    st.subheader("Equity Curve")
    if history:
        history_df = pd.DataFrame(history)
        history_df["ts"] = pd.to_datetime(history_df["ts"])
        history_df = history_df.set_index("ts")
        st.line_chart(history_df[["total_equity"]])
    else:
        st.info("Sin datos en portfolio_history para el rango elegido.")

    st.subheader("The Agent Debate")
    agent_rows = build_agent_rows(reports, asset_filter=asset_filter)
    if agent_rows:
        for row in agent_rows:
            header = (
                f"{row['agent']} | {row['signal']} | "
                f"conf: {row['confidence']}"
            )
            with st.expander(header, expanded=False):
                st.write(f"**Summary:** {row['summary']}")
                st.write(f"**Asset:** {row['asset']}")
                st.write(f"**Impacted Sectors:** {row['impacted_sectors']}")
                st.write(f"**Timestamp:** {row['timestamp']}")
    else:
        st.info("No hay reportes recientes de agentes.")

    st.subheader("Consensus Timeline")
    if decisions:
        decisions_df = pd.DataFrame(decisions)
        st.dataframe(
            decisions_df[["ts", "action", "consensus_score", "rationale"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hay decisiones registradas en final_decisions.")

    st.subheader("Open Positions")
    if positions:
        positions_df = pd.DataFrame(positions)
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hay posiciones abiertas.")

    st.subheader("Executed Trades")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["ts"] = pd.to_datetime(trades_df["ts"])
        pnl_by_symbol = (
            trades_df.groupby("symbol", as_index=False)["realized_pnl"].sum()
        )
        st.markdown("**PnL Realizado por Activo**")
        st.dataframe(
            pnl_by_symbol.sort_values("realized_pnl", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Ledger de Operaciones**")
        st.dataframe(
            trades_df[
                [
                    "ts",
                    "side",
                    "symbol",
                    "quantity",
                    "price",
                    "fee",
                    "notional",
                    "realized_pnl",
                    "source_signal",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hay trades ejecutados para los filtros elegidos.")

    st.caption(
        "Ultima actualizacion: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        "(auto-refresh 30s)"
    )


if hasattr(st, "experimental_fragment"):

    @st.experimental_fragment(run_every="30s")
    def auto_refresh_fragment() -> None:
        render_dashboard()

    auto_refresh_fragment()
else:
    render_dashboard()
