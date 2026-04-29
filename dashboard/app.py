"""Streamlit War Room dashboard for Lyntrix swarm and paper-trading state."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.database import Database  # noqa: E402
from agents.interface_agent import InterfaceAgent  # noqa: E402


ASSET_LABELS = {
    "HG=F": "Cobre",
    "GC=F": "Oro",
    "NLR": "Energia (NLR)",
}


def _ensure_page_config() -> None:
    if st.session_state.get("_warroom_page_cfg"):
        return
    st.set_page_config(
        page_title="Lyntrix War Room",
        layout="wide",
    )
    st.session_state["_warroom_page_cfg"] = True


def translate_signal(value: Any) -> str:
    """Translate common signal/action labels to Spanish."""
    token = str(value or "-").upper()
    mapping = {
        "BUY": "COMPRA",
        "STRONG_BUY": "COMPRA FUERTE",
        "SELL": "VENTA",
        "HOLD": "MANTENER",
        "ABORT/STAY_OUT": "ABORTAR / FUERA DEL MERCADO",
        "RISK_ON": "RIESGO ACTIVO",
        "RISK_OFF": "RIESGO DEFENSIVO",
        "INFO": "INFORMATIVO",
        "NEUTRAL": "NEUTRAL",
        "HIGH_IMPACT_EVENT": "ALTO IMPACTO",
    }
    return mapping.get(token, token)


def translate_regime(value: Any) -> str:
    """Translate macro regime label."""
    token = str(value or "UNKNOWN").upper()
    if token == "RISK_ON":
        return "RIESGO ACTIVO"
    if token == "RISK_OFF":
        return "RIESGO DEFENSIVO"
    return token


def to_es_summary(text: str) -> str:
    """Lightweight summary localization for key market terms."""
    translated = text
    replacements = {
        "risk-off": "riesgo defensivo",
        "risk on": "riesgo activo",
        "risk-on": "riesgo activo",
        "supply-chain": "cadena de suministro",
        "hard assets": "activos reales",
        "commodities": "materias primas",
        "Energy": "Energia",
        "Tech": "Tecnologia",
    }
    for src, dst in replacements.items():
        translated = translated.replace(src, dst).replace(src.title(), dst)
    return translated


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
                "summary_es": to_es_summary(summary),
                "signal": payload.get("signal", payload.get("action", "-")),
                "asset": asset,
                "impacted_sectors": impacted,
            }
        )
    return rows


def _parse_calendar_events(
    calendar_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten calendar_agent report rows for the macro strip."""
    out: list[dict[str, Any]] = []
    for row in calendar_rows:
        p = normalize_payload(row.get("payload"))
        minutes = p.get("minutes_to_event")
        if minutes is not None:
            try:
                minutes = int(minutes)
            except (TypeError, ValueError):
                minutes = None
        out.append(
            {
                "ts": row.get("ts"),
                "signal": str(p.get("signal", "INFO")),
                "event_name": str(p.get("event_name", "-")),
                "minutes_to_event": minutes,
                "risk_expected": str(p.get("risk_expected", "-")),
                "summary": str(p.get("summary", "")),
            }
        )
    return out


def _macro_countdown_html(events: list[dict[str, Any]]) -> str:
    """Build HTML/CSS strip: amber blink if high impact and < 60 min."""
    if not events:
        return (
            "<div style='padding:12px;border-radius:8px;"
            "background:#1e1e2e;color:#ccc;'>"
            "Macro Countdown: sin eventos persistidos del CalendarAgent."
            "</div>"
        )

    latest = max(events, key=lambda e: e.get("ts") or datetime.min)
    blink = (
        latest["signal"] == "HIGH_IMPACT_EVENT"
        and latest["minutes_to_event"] is not None
        and latest["minutes_to_event"] < 60
    )
    alert_cls = " warroom-macro-alert" if blink else ""

    style_block = """
    <style>
    @keyframes warroom-blink {
      0%, 100% { opacity: 1; filter: brightness(1); }
      50% { opacity: 0.72; filter: brightness(0.92); }
    }
    .warroom-macro-alert {
      animation: warroom-blink 1.1s ease-in-out infinite;
      background: linear-gradient(90deg, #b8860b22, #d4a01744) !important;
      border-color: #d4a017 !important;
      box-shadow: 0 0 18px #d4a01755;
    }
    .warroom-macro-strip {
      border: 1px solid #333;
      border-radius: 10px;
      padding: 14px 16px;
      background: #12121a;
      color: #eaeaf0;
      margin-bottom: 12px;
    }
    .warroom-macro-title { font-size: 0.85rem; color: #9aa0a6; margin-bottom: 8px; }
    .warroom-chip { display: inline-block; margin: 4px 8px 0 0; padding: 6px 10px;
      border-radius: 6px; background: #1f2430; font-size: 0.8rem; }
    </style>
    """

    chips = []
    for ev in sorted(
        events,
        key=lambda e: (
            e["minutes_to_event"]
            if e["minutes_to_event"] is not None
            else 10**9
        ),
    )[:5]:
        m = ev["minutes_to_event"]
        m_s = f"{m} min" if m is not None else "n/d"
        sig_es = translate_signal(ev["signal"])
        chips.append(
            f"<span class='warroom-chip'><b>{ev['event_name']}</b> — "
            f"{m_s} · {sig_es} · riesgo {ev['risk_expected']}</span>"
        )

    main = (
        f"<div class='warroom-macro-strip{alert_cls}'>"
        f"<div class='warroom-macro-title'>Macro Countdown "
        f"(CalendarAgent)</div>"
        f"<div><b>Proximo foco:</b> {latest.get('summary', '')}</div>"
        f"<div style='margin-top:8px;'>{''.join(chips)}</div>"
        "</div>"
    )
    return style_block + main


def _zscore_last(series: pd.Series) -> float | None:
    """Z = (x - mu) / sigma on the window; x = last close."""
    s = series.dropna().astype(float)
    if len(s) < 3:
        return None
    x = float(s.iloc[-1])
    mu = float(s.mean())
    sigma = float(s.std(ddof=0))
    if sigma < 1e-12:
        return None
    return (x - mu) / sigma


def build_zscore_heatmap_figure(zrows: list[dict[str, Any]]) -> go.Figure | None:
    """Heatmap of current Z-scores: green tail Z < -2, red Z > 2 (RdYlGn_r)."""
    if not zrows:
        return None
    df = pd.DataFrame(zrows)
    if df.empty or "symbol" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    z_map: dict[str, float | None] = {}
    for sym in ("HG=F", "GC=F", "NLR"):
        sub = df[df["symbol"] == sym].sort_values("ts")["close"]
        z_map[sym] = _zscore_last(sub)

    ratio_z: float | None = None
    hg = df[df["symbol"] == "HG=F"].sort_values("ts")[["ts", "close"]].rename(
        columns={"close": "hg"},
    )
    gc = df[df["symbol"] == "GC=F"].sort_values("ts")[["ts", "close"]].rename(
        columns={"close": "gc"},
    )
    if not hg.empty and not gc.empty:
        merged = pd.merge(hg, gc, on="ts", how="inner")
        merged = merged[(merged["hg"] > 0) & (merged["gc"] > 0)]
        if len(merged) >= 3:
            merged["ratio"] = merged["hg"] / merged["gc"]
            ratio_z = _zscore_last(merged["ratio"])

    z_map["Cu/Au"] = ratio_z

    labels = ["Cobre (HG=F)", "Oro (GC=F)", "Energia (NLR)", "Ratio Cu/Au"]
    keys = ["HG=F", "GC=F", "NLR", "Cu/Au"]
    values: list[float] = []
    text_cells: list[str] = []
    for k in keys:
        zv = z_map.get(k)
        if zv is None:
            values.append(0.0)
            text_cells.append("n/d")
        else:
            values.append(float(zv))
            text_cells.append(f"{zv:.2f}")

    z_plot = [values]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_plot,
            x=labels,
            y=["Z actual"],
            text=[text_cells],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="RdYlGn_r",
            zmin=-3,
            zmax=3,
            hovertemplate="%{x}<br>Z=%{z:.3f}<extra></extra>",
            colorbar=dict(title="Z"),
        ),
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=220,
        paper_bgcolor="#0e0e14",
        plot_bgcolor="#0e0e14",
        font=dict(color="#eaeaf0"),
        title=dict(
            text="Z-Score heatmap (ventana ~22 cierres diarios)",
            font=dict(size=14),
        ),
    )
    return fig


def build_whale_dual_axis_figure(
    candles: list[dict[str, Any]],
    whale_oi: list[dict[str, Any]],
) -> go.Figure | None:
    """Y1: Cobre/Oro closes; Y2: WhaleScout OI proxy (macro_indicators)."""
    if not candles:
        return None
    cdf = pd.DataFrame(candles)
    cdf["ts"] = pd.to_datetime(cdf["ts"])
    cdf["close"] = pd.to_numeric(cdf["close"], errors="coerce")

    fig = go.Figure()
    for sym, label, color in (
        ("HG=F", "Cobre (HG=F)", "#f39c12"),
        ("GC=F", "Oro (GC=F)", "#f1c40f"),
    ):
        sub = cdf[cdf["symbol"] == sym].sort_values("ts")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["ts"],
                y=sub["close"],
                name=label,
                mode="lines",
                line=dict(color=color, width=2),
                yaxis="y",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
            ),
        )

    if whale_oi:
        wdf = pd.DataFrame(whale_oi)
        wdf["ts"] = pd.to_datetime(wdf["ts"])
        wdf["value"] = pd.to_numeric(wdf["value"], errors="coerce")
        for ind, label, dash in (
            ("WHALE_OI::HG=F", "OI Cobre (proxy)", "dot"),
            ("WHALE_OI::GC=F", "OI Oro (proxy)", "dash"),
        ):
            sub = wdf[wdf["indicator"] == ind].sort_values("ts")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["ts"],
                    y=sub["value"],
                    name=label,
                    mode="lines",
                    line=dict(width=1.5, dash=dash),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
                ),
            )

    fig.update_layout(
        title=dict(text="Whale Activity: precio vs interes abierto (proxy)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(
            title=dict(text="Precio (cierre)"),
            side="left",
            showgrid=True,
            gridcolor="#2a2a36",
        ),
        yaxis2=dict(
            title=dict(text="Open Interest (WhaleScout)"),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        paper_bgcolor="#0e0e14",
        plot_bgcolor="#0e0e14",
        font=dict(color="#eaeaf0"),
        height=480,
        margin=dict(l=50, r=60, t=60, b=40),
    )
    return fig


async def fetch_dashboard_data(
    start_date: date,
    end_date: date,
    asset_filter: str,
    trades_limit: int,
) -> dict[str, Any]:
    """Bundle DB reads + executive briefing (InterfaceAgent)."""
    symbol_map = {
        "ALL": None,
        "COPPER": "HG=F",
        "GOLD": "GC=F",
        "ENERGY": "NLR",
    }
    trades_symbol = symbol_map[asset_filter]

    db = Database()
    try:
        data = await db.fetch_dashboard_bundle(
            start_date=start_date,
            end_date=end_date,
            trades_symbol=trades_symbol,
            trades_limit=trades_limit,
        )
    finally:
        await db.close()

    iface = InterfaceAgent()
    try:
        data["executive_briefing"] = await iface.executive_briefing()
    finally:
        await iface.close()

    macro = data.get("macro_regime")
    data["macro_regime"] = macro if macro else "UNKNOWN"
    bs = data.get("black_swan_risk")
    data["black_swan_risk"] = bs if bs else "LOW"
    return data


def render_dashboard() -> None:
    """Render one full dashboard refresh frame."""
    _ensure_page_config()
    st.session_state.setdefault("chat_history", [])

    st.title("Lyntrix War Room")

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
        trades_limit = st.slider(
            "Max trades",
            min_value=20,
            max_value=500,
            value=200,
        )
        st.caption(
            "Chat con Oracle: pestaña **Chat with Oracle** "
            "(historial persistente en esta sesión)."
        )

    data = asyncio.run(
        fetch_dashboard_data(
            start_date=start_date,
            end_date=end_date,
            asset_filter=asset_filter,
            trades_limit=trades_limit,
        )
    )

    snapshot = data["snapshot"]
    history = data["history"]
    reports = data["reports"]
    decisions = data["decisions"]
    positions = data["positions"]
    trades = data["trades"]
    candles = data["candles"]
    whale_oi = data.get("whale_oi") or []
    calendar_events_raw = data.get("calendar_events") or []
    zscore_rows = data.get("zscore_closes") or []
    macro_regime = data["macro_regime"]
    black_swan_risk = data["black_swan_risk"]
    executive_briefing = data.get("executive_briefing", "")

    cal_parsed = _parse_calendar_events(calendar_events_raw)
    st.markdown(_macro_countdown_html(cal_parsed), unsafe_allow_html=True)

    tab_main, tab_debate, tab_portfolio, tab_oracle = st.tabs(
        [
            "Main Dashboard",
            "Agent Debate",
            "Portfolio History",
            "Chat with Oracle",
        ],
    )

    with tab_main:
        if str(black_swan_risk).upper() == "CRITICAL":
            st.error(
                "BLACK SWAN ALERT: riesgo CRITICAL detectado. "
                "Modo defensivo activo (ABORT/STAY_OUT).",
                icon="🚨",
            )

        latest_decision = decisions[0] if decisions else None
        if latest_decision is not None:
            summary = str(
                latest_decision.get("consensus_logic")
                or latest_decision.get("rationale")
                or "Sin resumen ejecutivo disponible.",
            )
            st.success(
                (
                    f"Decision actual: {latest_decision.get('action', 'N/A')} | "
                    "Score: "
                    f"{float(latest_decision.get('consensus_score', 0.0)):.2f}\n\n"
                    f"{summary}"
                ),
                icon="🧠",
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

        total_trades = len(trades)
        realized_total = (
            float(pd.DataFrame(trades)["realized_pnl"].sum()) if trades else 0.0
        )
        open_positions_count = len(positions)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Balance Actual", f"${current_balance:,.2f}")
        m2.metric("PnL Total (%)", f"{pnl_pct:+.2f}%")
        m3.metric("Regimen Macro", translate_regime(macro_regime))
        m4.metric("Trades Ejecutados", f"{total_trades}")
        m5.metric("PnL Realizado", f"${realized_total:,.2f}")
        m6.metric("Posiciones Abiertas", f"{open_positions_count}")

        st.subheader("Deep Insight — Executive Briefing")
        st.container(border=True).markdown(
            executive_briefing.replace("\n", "\n\n")
            if executive_briefing
            else "_Sin briefing._",
        )

        st.subheader("Whale Activity Panel")
        whale_fig = build_whale_dual_axis_figure(candles, whale_oi)
        if whale_fig is not None:
            st.plotly_chart(whale_fig, use_container_width=True)
        else:
            st.info("Sin velas u OI persistido para armar el panel dual.")

        st.subheader("Z-Score heatmap")
        zfig = build_zscore_heatmap_figure(zscore_rows)
        if zfig is not None:
            st.plotly_chart(zfig, use_container_width=True)
            st.caption(
                r"Formula $Z = \frac{x - \mu}{\sigma}$ sobre ~22 cierres diarios. "
                "Escala divergente: colas verdes (Z bajo) vs rojas (Z alto); "
                "umbral visual ±2."
            )
        else:
            st.info("Sin datos suficientes para Z-score.")

        st.subheader("Panel de Mercado")
        if candles:
            candles_df = pd.DataFrame(candles)
            candles_df["ts"] = pd.to_datetime(candles_df["ts"])
            candles_df["asset_es"] = candles_df["symbol"].map(ASSET_LABELS).fillna(
                candles_df["symbol"],
            )
            pivot_close = candles_df.pivot_table(
                index="ts",
                columns="asset_es",
                values="close",
                aggfunc="last",
            ).sort_index()
            st.markdown("**Precios de referencia (Cierre)**")
            mkt = go.Figure()
            for col in pivot_close.columns:
                mkt.add_trace(
                    go.Scatter(
                        x=pivot_close.index,
                        y=pivot_close[col],
                        name=str(col),
                        mode="lines",
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
                    ),
                )
            mkt.update_layout(
                height=400,
                hovermode="x unified",
                paper_bgcolor="#0e0e14",
                plot_bgcolor="#0e0e14",
                font=dict(color="#eaeaf0"),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(mkt, use_container_width=True)

            returns = pivot_close.pct_change().fillna(0.0)
            cumulative = (1.0 + returns).cumprod() - 1.0
            st.markdown("**Rendimiento acumulado del periodo**")
            ret_fig = go.Figure()
            for col in cumulative.columns:
                ret_fig.add_trace(
                    go.Scatter(
                        x=cumulative.index,
                        y=cumulative[col],
                        name=str(col),
                        mode="lines",
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
                    ),
                )
            ret_fig.update_layout(
                height=360,
                hovermode="x unified",
                paper_bgcolor="#0e0e14",
                plot_bgcolor="#0e0e14",
                font=dict(color="#eaeaf0"),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(ret_fig, use_container_width=True)
        else:
            st.info(
                "Sin velas en `candles` para graficar mercado "
                "en el rango elegido.",
            )

        st.subheader("Equity Curve")
        if history:
            history_df = pd.DataFrame(history)
            history_df["ts"] = pd.to_datetime(history_df["ts"])
            eq = go.Figure(
                data=go.Scatter(
                    x=history_df["ts"],
                    y=history_df["total_equity"],
                    name="Total equity",
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#3498db"),
                    hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
                ),
            )
            eq.update_layout(
                height=360,
                paper_bgcolor="#0e0e14",
                plot_bgcolor="#0e0e14",
                font=dict(color="#eaeaf0"),
            )
            st.plotly_chart(eq, use_container_width=True)
        else:
            st.info("Sin datos en portfolio_history para el rango elegido.")

        st.subheader("Open Positions")
        if positions:
            positions_df = pd.DataFrame(positions)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay posiciones abiertas.")

        st.subheader("Comunicacion del Swarm (ES)")
        agent_rows_main = build_agent_rows(reports, asset_filter=asset_filter)
        if agent_rows_main:
            swarm_rows = [
                {
                    "Agente": row["agent"],
                    "Senal": translate_signal(row["signal"]),
                    "Activo": row["asset"],
                    "Confianza": row["confidence"],
                    "Mensaje": row["summary_es"],
                    "Timestamp": row["timestamp"],
                }
                for row in agent_rows_main
            ]
            st.dataframe(
                pd.DataFrame(swarm_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Sin mensajes recientes para traducir.")

    with tab_debate:
        st.subheader("The Agent Debate")
        agent_rows = build_agent_rows(reports, asset_filter=asset_filter)
        if agent_rows:
            for row in agent_rows:
                header = (
                    f"{row['agent']} | {translate_signal(row['signal'])} | "
                    f"conf: {row['confidence']}"
                )
                with st.expander(header, expanded=False):
                    st.write(f"**Summary:** {row['summary']}")
                    st.write(f"**Resumen (ES):** {row['summary_es']}")
                    st.write(f"**Asset:** {row['asset']}")
                    st.write(f"**Impacted Sectors:** {row['impacted_sectors']}")
                    st.write(f"**Timestamp:** {row['timestamp']}")
        else:
            st.info("No hay reportes recientes de agentes.")

        st.subheader("Agent Debate Logs (rationale humano)")
        if reports:
            for report in reports:
                rationale = str(report.get("human_rationale") or "").strip()
                if not rationale:
                    continue
                payload = normalize_payload(report.get("payload"))
                signal = payload.get("signal", payload.get("action", "-"))
                agent_name = report.get("agent", "-")
                title = f"{agent_name} | {translate_signal(signal)}"
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(f"{report.get('ts')}")
                    st.write(rationale)
        else:
            st.info("No hay rationale humano persistido todavia.")

        st.subheader("Consensus Timeline")
        if decisions:
            decisions_df = pd.DataFrame(decisions)
            decisions_df["action_es"] = decisions_df["action"].map(
                translate_signal,
            )
            decisions_df["ts"] = pd.to_datetime(decisions_df["ts"])
            score_series = decisions_df.sort_values("ts").set_index("ts")[
                ["consensus_score"]
            ]
            st.markdown("**Evolucion del score de consenso**")
            sc = go.Figure(
                data=go.Scatter(
                    x=score_series.index,
                    y=score_series["consensus_score"],
                    mode="lines+markers",
                    name="Score",
                    hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                ),
            )
            sc.update_layout(
                height=320,
                paper_bgcolor="#0e0e14",
                plot_bgcolor="#0e0e14",
                font=dict(color="#eaeaf0"),
            )
            st.plotly_chart(sc, use_container_width=True)
            st.dataframe(
                decisions_df[
                    [
                        "ts",
                        "action",
                        "action_es",
                        "consensus_score",
                        "rationale",
                        "consensus_logic",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No hay decisiones registradas en final_decisions.")

    with tab_portfolio:
        st.subheader("Equity Curve")
        if history:
            history_df = pd.DataFrame(history)
            history_df["ts"] = pd.to_datetime(history_df["ts"])
            eq2 = go.Figure(
                data=go.Scatter(
                    x=history_df["ts"],
                    y=history_df["total_equity"],
                    name="Total equity",
                    mode="lines",
                    line=dict(color="#2ecc71"),
                    hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
                ),
            )
            eq2.update_layout(
                height=400,
                paper_bgcolor="#0e0e14",
                plot_bgcolor="#0e0e14",
                font=dict(color="#eaeaf0"),
            )
            st.plotly_chart(eq2, use_container_width=True)
        else:
            st.info("Sin datos en portfolio_history para el rango elegido.")

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

    with tab_oracle:
        st.subheader("Chat with Oracle (InterfaceAgent)")
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input(
            "Preguntale al War Room (ej: que pasa con el oro?)",
            key="oracle_chat_input",
        )
        if user_prompt:
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_prompt},
            )
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analizando contexto historico del enjambre..."):
                    agent = InterfaceAgent()
                    try:
                        answer = asyncio.run(agent.answer(user_prompt))
                    finally:
                        asyncio.run(agent.close())
                st.markdown(answer)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer},
            )

    st.caption(
        "Ultima actualizacion: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        "(auto-refresh 30s en fragmento)"
    )


if hasattr(st, "experimental_fragment"):

    @st.experimental_fragment(run_every="30s")
    def auto_refresh_fragment() -> None:
        render_dashboard()

    auto_refresh_fragment()
else:
    render_dashboard()
