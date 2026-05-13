"""Conversational interface agent with DB-grounded RAG responses."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import groq
from groq import AsyncGroq

from shared.database import Database

LOGGER = logging.getLogger(__name__)


class InterfaceAgent:
    """Answer user questions from swarm-collected evidence."""

    ASSET_KEYWORDS = {
        "oro": "GC=F",
        "gold": "GC=F",
        "cobre": "HG=F",
        "copper": "HG=F",
        "energia": "NLR",
        "energy": "NLR",
        "nlr": "NLR",
    }

    def __init__(self) -> None:
        self._db = Database()
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self._client: AsyncGroq | None = None
        if self._groq_api_key:
            self._client = AsyncGroq(api_key=self._groq_api_key)

    async def close(self) -> None:
        """Release DB engine resources."""
        await self._db.close()

    async def executive_briefing(self) -> str:
        """Auto-generated 3-line executive summary from latest swarm evidence."""
        reports = await self._db.fetch_reports_for_query(
            keywords=[],
            asset_symbols=[],
            limit=8,
        )
        decisions = await self._db.fetch_recent_final_decisions(limit=3)
        portfolio = await self._db.fetch_recent_portfolio_snapshots(limit=5)
        if not reports and not decisions and not portfolio:
            return (
                "Sin datos persistidos suficientes para un briefing ejecutivo. "
                "Levanta agentes y espera el primer ciclo de reportes."
            )
        context = (
            "Genera un Executive Briefing de EXACTAMENTE 3 renglones "
            "(3 frases cortas separadas por salto de linea). "
            "Espanol, tono senior desk, solo datos del contexto. "
            "Sin markdown.\n\n"
            f"Reportes:\n{json.dumps(reports, ensure_ascii=True, default=str)}\n\n"
            f"Decisiones:\n{json.dumps(decisions, ensure_ascii=True, default=str)}\n\n"
            f"Portfolio:\n{json.dumps(portfolio, ensure_ascii=True, default=str)}\n"
        )
        if self._client is None:
            return self._briefing_fallback(decisions, portfolio)
        try:
            completion = await self._client.chat.completions.create(
                model=self._groq_model,
                temperature=0.2,
                max_tokens=220,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sos el CIO adjunto del War Room Lyntrix. "
                            "Salida: exactamente 3 lineas de texto plano."
                        ),
                    },
                    {"role": "user", "content": context},
                ],
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
        except Exception:
            LOGGER.exception("executive_briefing LLM failed")
        return self._briefing_fallback(decisions, portfolio)

    def _briefing_fallback(
        self,
        decisions: list[dict[str, Any]],
        portfolio: list[dict[str, Any]],
    ) -> str:
        """Three-line deterministic briefing."""
        line1 = "Contexto macro/tactico aun sin sintesis LLM (fallback)."
        if decisions:
            d0 = decisions[0]
            line1 = (
                f"Ultima decision: {d0.get('action')} "
                f"(score {d0.get('consensus_score')})."
            )
        line2 = "Monitorear calendario y riesgo cola antes de escalar size."
        if portfolio:
            p0 = portfolio[0]
            line2 = (
                f"Equity paper {p0.get('total_equity')} "
                f"(PnL real {p0.get('realized_pnl')})."
            )
        line3 = "Mantener disciplina hasta alinear Value, Macro y Whale."
        return f"{line1}\n{line2}\n{line3}"

    async def answer(self, question: str) -> str:
        """Resolve user question against DB evidence and synthesize answer."""
        cleaned = question.strip()
        if not cleaned:
            return (
                "Necesito una pregunta concreta para analizar "
                "el estado del sistema."
            )

        query_terms, asset_symbols = self._extract_focus(cleaned)
        reports = await self._db.fetch_reports_for_query(
            keywords=query_terms,
            asset_symbols=asset_symbols,
            limit=10,
        )
        decisions = await self._db.fetch_recent_final_decisions(limit=5)
        portfolio = await self._db.fetch_recent_portfolio_snapshots(limit=10)

        if not reports and not decisions and not portfolio:
            return (
                "No hay datos historicos suficientes en la base para responder "
                "esa consulta por ahora."
            )

        context = self._build_context(cleaned, reports, decisions, portfolio)
        if self._client is None:
            return self._heuristic_response(
                cleaned,
                reports,
                decisions,
                portfolio,
            )

        try:
            completion = await self._client.chat.completions.create(
                model=self._groq_model,
                temperature=0.15,
                max_tokens=450,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sos un Asistente Senior de Inversion para Lyntrix. "
                            "Responde en espanol, tecnico y preciso. "
                            "Basate SOLO en la evidencia del contexto. "
                            "Si falta evidencia, decilo explicitamente."
                        ),
                    },
                    {"role": "user", "content": context},
                ],
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
        except groq.RateLimitError:
            LOGGER.warning(
                "InterfaceAgent rate limited; using heuristic fallback",
            )
        except Exception:
            LOGGER.exception("InterfaceAgent LLM query failed")
        return self._heuristic_response(
            cleaned,
            reports,
            decisions,
            portfolio,
        )

    def _extract_focus(self, question: str) -> tuple[list[str], list[str]]:
        """Extract keyword terms and asset symbols from user question."""
        lowered = question.lower()
        symbols: set[str] = set()
        for token, symbol in self.ASSET_KEYWORDS.items():
            if token in lowered:
                symbols.add(symbol)

        words = re.findall(r"[a-zA-Z0-9=]{3,}", lowered)
        stopwords = {
            "que",
            "como",
            "para",
            "con",
            "los",
            "las",
            "del",
            "por",
            "una",
            "sobre",
            "esta",
            "este",
            "hoy",
        }
        terms = [word for word in words if word not in stopwords][:12]
        return terms, sorted(symbols)

    def _build_context(
        self,
        question: str,
        reports: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        portfolio: list[dict[str, Any]],
    ) -> str:
        """Serialize structured context for the model."""
        return (
            f"Pregunta del usuario: {question}\n\n"
            "Reportes relevantes (max 10):\n"
            f"{json.dumps(reports, ensure_ascii=True)}\n\n"
            "Decisiones recientes (max 5):\n"
            f"{json.dumps(decisions, ensure_ascii=True)}\n\n"
            "Snapshots de portfolio (max 10):\n"
            f"{json.dumps(portfolio, ensure_ascii=True)}\n"
        )

    def _heuristic_response(
        self,
        question: str,
        reports: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        portfolio: list[dict[str, Any]],
    ) -> str:
        """Fallback deterministic response grounded on available data."""
        del question
        lines: list[str] = []
        if decisions:
            latest = decisions[0]
            lines.append(
                "Decision mas reciente: "
                f"{latest.get('action')} "
                f"(score={latest.get('consensus_score')}).",
            )
        if portfolio:
            latest_pf = portfolio[0]
            lines.append(
                "Portfolio actual estimado: "
                f"equity={latest_pf.get('total_equity')}, "
                f"cash={latest_pf.get('cash_balance')}.",
            )
        if reports:
            agents = ", ".join(
                sorted({str(report.get("agent")) for report in reports}),
            )
            lines.append(f"Agentes con evidencia reciente: {agents}.")
        if not lines:
            return (
                "No encontre evidencia suficiente en reportes, decisiones o "
                "portfolio para contestar con precision."
            )
        lines.append(
            "Nota: respuesta en modo fallback por "
            "indisponibilidad temporal del LLM.",
        )
        return " ".join(lines)
