"""LLM-backed narrative layer for human-readable swarm reasoning."""

from __future__ import annotations

from collections import deque
import json
import logging
import os
from typing import Any

import groq
from groq import AsyncGroq

LOGGER = logging.getLogger(__name__)


class RationaleAgent:
    """Generate concise executive narratives from technical swarm reports."""

    def __init__(self) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=3)
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self._client: AsyncGroq | None = None
        if self._groq_api_key:
            self._client = AsyncGroq(api_key=self._groq_api_key)

    async def generate(
        self,
        incoming_report: dict[str, Any],
        latest_signals: dict[str, dict[str, Any]],
    ) -> str:
        """Create a two-paragraph max rationale with continuity context."""
        self._history.append(
            {
                "agent": incoming_report.get("agent"),
                "signal": incoming_report.get("signal"),
                "summary": incoming_report.get(
                    "summary",
                    incoming_report.get("reason"),
                ),
                "ts": incoming_report.get("timestamp"),
            }
        )
        context = self._build_context(incoming_report, latest_signals)
        if self._client is None:
            return self._heuristic_rationale(context)

        try:
            completion = await self._client.chat.completions.create(
                model=self._groq_model,
                temperature=0.25,
                max_tokens=350,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an elite buy-side analyst for Lyntrix War Room. "
                            "Write in Spanish, professional and direct tone. "
                            "Output plain text only, max 2 short paragraphs."
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
                "RationaleAgent hit rate limit; using heuristic text",
            )
        except Exception:
            LOGGER.exception("RationaleAgent LLM generation failed")
        return self._heuristic_rationale(context)

    def _build_context(
        self,
        incoming_report: dict[str, Any],
        latest_signals: dict[str, dict[str, Any]],
    ) -> str:
        """Build compact context for LLM prompt with rolling memory."""
        history_json = json.dumps(list(self._history), ensure_ascii=True)
        current_reports = {
            agent: node.get("payload", {})
            for agent, node in latest_signals.items()
        }
        current_json = json.dumps(current_reports, ensure_ascii=True)
        incoming_json = json.dumps(incoming_report, ensure_ascii=True)
        return (
            "Objetivo: explicar en lenguaje humano la situacion actual del swarm.\n"
            "Conecta geopolitica + macro + tecnica. "
            "Menciona continuidad temporal si aplica "
            "(ej: 'sigue cayendo pero ahora...').\n"
            "Si hay WhaleScout en acumulacion, incluye literalmente: "
            "'Veo que los institucionales estan entrando'.\n"
            "Si CalendarAgent marca evento de alto impacto cercano, incluye "
            "una advertencia tipo: 'Ojo con el anuncio de la FED en X minutos'.\n"
            "Maximo 2 parrafos.\n"
            f"Reporte entrante: {incoming_json}\n"
            f"Estado actual de agentes: {current_json}\n"
            f"Memoria ultimos 3 reportes: {history_json}\n"
        )

    def _heuristic_rationale(self, context: str) -> str:
        """Fast fallback rationale to avoid blocking ingestion path."""
        has_whale = "INSTITUTIONAL_ACCUMULATION" in context
        fed_minutes = self._extract_fed_minutes(context)
        extra_1 = ""
        extra_2 = ""
        if has_whale:
            extra_1 = " Veo que los institucionales estan entrando."
        if fed_minutes is not None and fed_minutes <= 60:
            extra_2 = f" Ojo con el anuncio de la FED en {fed_minutes} minutos."
        return (
            "El enjambre mantiene un sesgo prudente: la lectura macro reciente "
            "se sostiene en modo defensivo y limita la agresividad de entradas "
            "tacticas, aun cuando aparecen focos selectivos en energia "
            f"y materias primas.{extra_1}{extra_2}\n\n"
            "La continuidad de reportes confirma que no hay una divergencia "
            "suficiente para escalar riesgo de cartera; "
            "la prioridad operativa sigue siendo preservar capital "
            "mientras esperamos confirmaciones "
            "cruzadas mas robustas."
        )

    def _extract_fed_minutes(self, context: str) -> int | None:
        """Extract minutes-to-event from serialized context when present."""
        marker = '"minutes_to_event":'
        if marker not in context:
            return None
        chunk = context.split(marker, maxsplit=1)[1][:16]
        digits = "".join(char for char in chunk if char.isdigit())
        if not digits:
            return None
        return int(digits)
