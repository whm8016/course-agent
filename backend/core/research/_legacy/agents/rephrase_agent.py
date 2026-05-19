"""RephraseAgent - Topic rephrasing Agent (faithful port from DeepTutor)."""

from __future__ import annotations

from typing import Any

from ..base_agent import BaseAgent
from ..trace import build_trace_metadata, new_call_id
from ..utils.json_utils import extract_json_from_text


class RephraseAgent(BaseAgent):
    """Topic rephrasing Agent"""

    _MODE_TO_STYLE = {
        "notes": "study_notes",
        "report": "report",
        "comparison": "comparison",
        "learning_path": "learning_path",
    }

    @staticmethod
    def _build_trace_meta(iteration: int) -> dict[str, Any]:
        return build_trace_metadata(
            call_id=new_call_id("research-rephrase"),
            phase="rephrasing",
            label="Rephrase topic",
            call_kind="llm_generation",
            trace_role="thought",
            trace_kind="llm_generation",
            iteration=iteration,
        )

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
    ):
        language = config.get("system", {}).get("language", "zh")
        super().__init__(
            module_name="research",
            agent_name="rephrase_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )
        self.conversation_history: list[dict[str, Any]] = []
        self.session_history: list[dict[str, Any]] = config.get("conversation_history", [])
        intent_mode = str(config.get("intent", {}).get("mode", "") or "")
        reporting_style = str(config.get("reporting", {}).get("style", "") or "")
        self._research_style = reporting_style or self._MODE_TO_STYLE.get(intent_mode, "report")

    def reset_history(self) -> None:
        self.conversation_history = []

    def _get_mode_contract(self, stage: str) -> str:
        return (
            self.get_prompt("mode_contracts", f"{self._research_style}_{stage}", "") or ""
        ).strip()

    def _format_conversation_history(self) -> str:
        if not self.conversation_history:
            return ""

        parts = []
        for entry in self.conversation_history:
            role = entry.get("role", "unknown")
            iteration = entry.get("iteration", 0)
            content = entry.get("content", "")
            if role == "user":
                label = "[User - Initial Input]" if iteration == 0 else f"[User - Feedback (Round {iteration})]"
                parts.append(f"{label}\n{content}")
            elif role == "assistant":
                topic = content.get("topic", "") if isinstance(content, dict) else str(content)
                parts.append(f"[Assistant - Rephrased Topic (Round {iteration})]\n{topic}")
        return "\n\n".join(parts)

    async def process(
        self,
        user_input: str,
        iteration: int = 0,
        previous_result: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
    ) -> dict[str, Any]:
        print(f"\n{'=' * 70}")
        print(f"RephraseAgent - Topic Rephrasing (Iteration {iteration})")
        print(f"{'=' * 70}")

        if iteration == 0:
            self.reset_history()
            print(f"Original Input: {user_input}\n")
        else:
            print(f"User Feedback: {user_input}\n")

        self.conversation_history.append({"role": "user", "content": user_input, "iteration": iteration})

        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError(
                "RephraseAgent missing system prompt (system.role in rephrase_agent.yaml)"
            )
        if self.session_history:
            ctx_parts = []
            for msg in self.session_history:
                role = msg.get("role", "user")
                content = str(msg.get("content", "")).strip()
                if content:
                    ctx_parts.append(f"[{role}]: {content}")
            if ctx_parts:
                system_prompt += (
                    "\n\n<session_history>\n"
                    "The following is the earlier conversation in this session.\n\n"
                    + "\n\n".join(ctx_parts)
                    + "\n</session_history>"
                )

        user_prompt_template = self.get_prompt("process", "rephrase")
        if not user_prompt_template:
            raise ValueError(
                "RephraseAgent missing rephrase prompt (process.rephrase in rephrase_agent.yaml)"
            )

        history_text = self._format_conversation_history()
        user_prompt = user_prompt_template.format(
            user_input=user_input,
            iteration=iteration,
            conversation_history=history_text,
            previous_result=history_text,
            mode_instruction=self._get_mode_contract("rephrase"),
        )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="rephrase",
            attachments=attachments,
            trace_meta=self._build_trace_meta(iteration),
        ):
            _chunks.append(_c)
        response = "".join(_chunks)

        data = extract_json_from_text(response)
        from ..utils.json_utils import ensure_json_dict, ensure_keys

        try:
            result = ensure_json_dict(data)
            ensure_keys(result, ["topic"])
        except Exception:
            fallback_topic = user_input
            for entry in reversed(self.conversation_history):
                if entry.get("role") == "assistant":
                    c = entry.get("content", {})
                    if isinstance(c, dict) and c.get("topic"):
                        fallback_topic = c["topic"]
                        break
            result = {"topic": fallback_topic}

        result["iteration"] = iteration
        self.conversation_history.append({"role": "assistant", "content": result, "iteration": iteration})
        print(f"\nRephrasing completed: {result.get('topic', '')}")
        return result


__all__ = ["RephraseAgent"]
