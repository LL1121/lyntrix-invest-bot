"""Consensus orchestrator skeleton for Lyntrix swarm reports."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from agents.rationale_agent import RationaleAgent
from shared.database import AgentReportRecord, Database, FinalDecisionRecord
from shared.messaging import MessageBus

LOGGER = logging.getLogger(__name__)
AgentName = Literal[
    "geo_oracle",
    "value_hunter",
    "macro_scout",
    "black_swan",
    "whale_scout",
    "calendar_agent",
]


class Orchestrator:
    """Consensus engine that aggregates swarm reports into execution signals."""

    REPORT_CHANNEL = "agents.reports"
    EXECUTION_CHANNEL = "execution.signals"
    GRACE_PERIOD_MINUTES = 10
    MIN_ACTIVE_AGENTS = 3

    def __init__(self, bus: MessageBus | None = None) -> None:
        self.bus = bus or MessageBus()
        self.db = Database()
        self.logger = logging.getLogger(__name__)
        self.latest_signals: dict[str, dict[str, Any]] = {}
        self._last_action: str | None = None
        self._state_lock = asyncio.Lock()
        self.rationale_agent = RationaleAgent()
        self._weights: dict[str, float] = {
            "macro_scout": 0.40,
            "value_hunter": 0.35,
            "geo_oracle": 0.25,
        }

    async def run(self) -> None:
        """Consume reports, compute consensus, and emit execution signals."""
        await self.bus.connect()
        await self.db.init_hypertables()
        pubsub = await self.bus.subscribe([self.REPORT_CHANNEL])
        self.logger.info(
            "Orchestrator listening on channel: %s",
            self.REPORT_CHANNEL,
        )

        try:
            async for message in self.bus.iter_messages(pubsub):
                payload = message["payload"]
                task = asyncio.create_task(self._process_report(payload))
                task.add_done_callback(self._log_task_exception)
        except asyncio.CancelledError:
            self.logger.info("Orchestrator cancelled")
            raise
        except Exception:
            self.logger.exception(
                "Orchestrator failed while processing events",
            )
            raise
        finally:
            await pubsub.aclose()
            await self.db.close()
            await self.bus.close()

    async def _process_report(self, payload: dict[str, Any]) -> None:
        """Update latest signal state and evaluate consensus."""
        agent = str(payload.get("agent", "")).lower()
        if not agent:
            self.logger.warning("Skipping report without agent field: %s", payload)
            return

        async with self._state_lock:
            self.latest_signals[agent] = {
                "payload": payload,
                "received_at": datetime.now(UTC),
            }
            rationale_text = await self._generate_rationale(payload)
            decision = self._compute_consensus()

        await self.db.insert_agent_report(
            AgentReportRecord(
                ts=datetime.now(UTC),
                agent=agent,
                payload=payload,
                human_rationale=rationale_text,
            )
        )
        self.logger.info("Signal updated | agent=%s payload=%s", agent, payload)

        if decision is None:
            return

        action = decision["action"]
        if action == self._last_action:
            return
        self._last_action = action

        output_payload = {
            "agent": "orchestrator",
            "action": action,
            "consensus_score": decision["consensus_score"],
            "votes_for": decision["votes_for"],
            "rationale": decision["rationale"],
            "executive_summary": decision["executive_summary"],
            "consensus_logic": decision["consensus_logic"],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self.bus.publish(self.EXECUTION_CHANNEL, output_payload)
        await self.db.insert_final_decision(
            FinalDecisionRecord(
                ts=datetime.now(UTC),
                action=action,
                consensus_score=float(decision["consensus_score"]),
                votes_for=decision["votes_for"],
                latest_signals=decision["signals_snapshot"],
                rationale=decision["rationale"],
                consensus_logic=decision["consensus_logic"],
            )
        )
        self.logger.info("Execution signal emitted: %s", output_payload)

    def _log_task_exception(self, task: asyncio.Task[Any]) -> None:
        """Surface background processing exceptions in orchestrator logs."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.exception("Background report task failed", exc_info=exc)

    async def _generate_rationale(self, payload: dict[str, Any]) -> str:
        """Generate human-readable rationale without stalling indefinitely."""
        try:
            return await asyncio.wait_for(
                self.rationale_agent.generate(payload, self.latest_signals),
                timeout=7,
            )
        except TimeoutError:
            self.logger.warning("Rationale generation timeout for agent=%s", payload.get("agent"))
            return (
                "No se pudo completar el rationale en tiempo objetivo; "
                "se mantiene el flujo tecnico con prioridad de latencia."
            )

    def _compute_consensus(self) -> dict[str, Any] | None:
        """Return weighted consensus decision if grace period is satisfied."""
        fresh_signals = self._fresh_agent_signals()
        if len(fresh_signals) < self.MIN_ACTIVE_AGENTS:
            self.logger.info(
                "Grace period active: %s/%s fresh agents in last %s min",
                len(fresh_signals),
                self.MIN_ACTIVE_AGENTS,
                self.GRACE_PERIOD_MINUTES,
            )
            return None

        black_swan = fresh_signals.get("black_swan", {}).get("payload", {})
        if str(black_swan.get("risk_level", "")).upper() == "CRITICAL":
            rationale = "BlackSwan veto active: CRITICAL risk"
            self.logger.warning("Consenso: 0.00 [Macro: N/A, Value: N/A, Geo: N/A]")
            return {
                "action": "ABORT/STAY_OUT",
                "consensus_score": 0.0,
                "votes_for": [],
                "rationale": rationale,
                "executive_summary": (
                    "Modo preservacion total: BlackSwan reporta riesgo critico y "
                    "se cancela toda exposicion tactica."
                ),
                "consensus_logic": (
                    "Veto absoluto de BlackSwan (CRITICAL) => score forzado a 0.0, "
                    "accion ABORT/STAY_OUT."
                ),
                "signals_snapshot": self._signals_snapshot(fresh_signals),
            }

        macro_payload = fresh_signals.get("macro_scout", {}).get("payload", {})
        value_payload = fresh_signals.get("value_hunter", {}).get("payload", {})
        geo_payload = fresh_signals.get("geo_oracle", {}).get("payload", {})
        whale_payload = fresh_signals.get("whale_scout", {}).get("payload", {})
        calendar_payload = fresh_signals.get("calendar_agent", {}).get("payload", {})

        macro_vote = self._macro_vote(macro_payload)
        value_vote = self._value_vote(value_payload)
        geo_vote = self._geo_vote(geo_payload)
        whale_bonus = self._whale_bonus(whale_payload, value_payload)
        score = (
            (macro_vote * self._weights["macro_scout"])
            + (value_vote * self._weights["value_hunter"])
            + (geo_vote * self._weights["geo_oracle"])
        )
        score += whale_bonus
        score = self._apply_calendar_precaution(
            score=score,
            calendar_payload=calendar_payload,
            macro_vote=macro_vote,
            value_vote=value_vote,
            geo_vote=geo_vote,
        )
        score = max(-1.0, min(1.0, score))
        action = self._score_to_action(score)
        votes_for = [name for name, vote in {
            "macro_scout": macro_vote,
            "value_hunter": value_vote,
            "geo_oracle": geo_vote,
        }.items() if vote > 0.0]

        macro_tag = "OK" if macro_vote > 0 else "WARN"
        value_tag = "OK" if value_vote > 0 else ("WARN" if value_vote < 0 else "NEUTRAL")
        geo_tag = "OK" if geo_vote > 0 else ("WARN" if geo_vote < 0 else "NEUTRAL")
        self.logger.info(
            "Consenso: %.2f [Macro: %s, Value: %s, Geo: %s, WhaleBonus: %.2f]",
            score,
            macro_tag,
            value_tag,
            geo_tag,
            whale_bonus,
        )
        return {
            "action": action,
            "consensus_score": score,
            "votes_for": votes_for,
            "rationale": (
                "Weighted blend of macro/value/geo signals with "
                "black-swan guardrails."
            ),
            "executive_summary": (
                f"Decision {action}: macro={macro_tag}, value={value_tag}, "
                f"geo={geo_tag}, whale_bonus={whale_bonus:.2f}, score={score:.2f}. "
                "Se prioriza disciplina de riesgo y confirmacion cruzada."
            ),
            "consensus_logic": (
                "Score = macro_vote*0.40 + value_vote*0.35 + geo_vote*0.25; "
                f"macro_vote={macro_vote:.3f}, value_vote={value_vote:.3f}, "
                f"geo_vote={geo_vote:.3f}, whale_bonus={whale_bonus:.3f}, "
                f"calendar_minutes={calendar_payload.get('minutes_to_event')}, "
                f"final_score={score:.3f}, action={action}."
            ),
            "signals_snapshot": self._signals_snapshot(fresh_signals),
        }

    def _fresh_agent_signals(self) -> dict[str, dict[str, Any]]:
        """Filter agent reports by grace-period freshness."""
        cutoff = datetime.now(UTC) - timedelta(minutes=self.GRACE_PERIOD_MINUTES)
        return {
            name: report
            for name, report in self.latest_signals.items()
            if report["received_at"] >= cutoff
        }

    def _signals_snapshot(
        self,
        signals: dict[str, dict[str, Any]],
    ) -> dict[str, object]:
        """Serialize only payloads for persistence."""
        return {name: data["payload"] for name, data in signals.items()}

    def _macro_vote(self, payload: dict[str, Any]) -> float:
        """Convert macro regime to weighted directional vote."""
        if not payload:
            return 0.0
        regime = str(payload.get("macro_regime", "")).upper()
        confidence = float(payload.get("confidence", 0.5))
        if regime == "RISK_OFF":
            return -1.0 * confidence
        if regime == "RISK_ON":
            return 0.8 * confidence
        return 0.0

    def _value_vote(self, payload: dict[str, Any]) -> float:
        """Convert ValueHunter discrete signal to numeric vote."""
        if not payload:
            return 0.0
        signal = str(payload.get("signal", "")).upper()
        confidence = float(payload.get("confidence", 0.5))
        mapper = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
        return mapper.get(signal, 0.0) * confidence

    def _geo_vote(self, payload: dict[str, Any]) -> float:
        """Use GeoOracle sentiment in [-1, 1] as vote."""
        if not payload:
            return 0.0
        sentiment = float(payload.get("sentiment_score", 0.0))
        return max(-1.0, min(1.0, sentiment))

    def _score_to_action(self, score: float) -> str:
        """Map consensus score to orchestrator action."""
        if score >= 0.7:
            return "STRONG_BUY"
        if score >= 0.25:
            return "BUY"
        if score <= -0.5:
            return "SELL"
        return "HOLD"

    def _whale_bonus(
        self,
        whale_payload: dict[str, Any],
        value_payload: dict[str, Any],
    ) -> float:
        """Add confidence bonus when WhaleScout confirms ValueHunter direction."""
        whale_signal = str(whale_payload.get("signal", "")).upper()
        value_signal = str(value_payload.get("signal", "")).upper()
        whale_conf = float(whale_payload.get("confidence", 0.0))
        if whale_signal == "INSTITUTIONAL_ACCUMULATION" and value_signal == "BUY":
            return min(0.15, 0.05 + whale_conf * 0.10)
        return 0.0

    def _apply_calendar_precaution(
        self,
        score: float,
        calendar_payload: dict[str, Any],
        macro_vote: float,
        value_vote: float,
        geo_vote: float,
    ) -> float:
        """Halve score when high-impact event is under 1h unless unanimous."""
        minutes = calendar_payload.get("minutes_to_event")
        try:
            minutes_value = int(minutes)
        except (TypeError, ValueError):
            return score
        if minutes_value > 60:
            return score
        if self._is_unanimous_signal(macro_vote, value_vote, geo_vote):
            return score
        self.logger.warning(
            "Calendar precaution active: event in %s min, score halved",
            minutes_value,
        )
        return score * 0.5

    def _is_unanimous_signal(self, macro: float, value: float, geo: float) -> bool:
        """True when all three core votes share same non-zero direction."""
        votes = [macro, value, geo]
        if any(abs(vote) <= 1e-9 for vote in votes):
            return False
        all_positive = all(vote > 0.0 for vote in votes)
        all_negative = all(vote < 0.0 for vote in votes)
        return all_positive or all_negative


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def main() -> None:
    configure_logging()
    orchestrator = Orchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Orchestrator interrupted by user")
