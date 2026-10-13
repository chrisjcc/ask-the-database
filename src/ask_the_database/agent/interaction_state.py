"""Multi-turn interaction state tracking for database exploration agents."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    """Record of a single turn in an agentic trajectory."""

    turn_index: int
    role: str
    action: str | None = None
    action_input: Any | None = None
    observation: str = ""
    is_terminal: bool = False
    latency_ms: float = 0.0
    had_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert turn record to dictionary."""
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "is_terminal": self.is_terminal,
            "latency_ms": self.latency_ms,
            "had_error": self.had_error,
        }


@dataclass
class InteractionState:
    """Full interaction state tracking trajectory history, errors, and terminal answers."""

    question: str
    db_id: str
    db_path: str
    gold_sql: str | None = None
    max_turns: int = 5
    turns: list[TurnRecord] = field(default_factory=list)

    def add_turn(
        self,
        action: str | None,
        action_input: Any | None,
        observation: str,
        is_terminal: bool = False,
        latency_ms: float = 0.0,
    ) -> TurnRecord:
        """Record an assistant turn and its corresponding observation."""
        turn_idx = len(self.turns) + 1
        had_error = (
            "Database Error" in observation
            or "Error:" in observation
            or "Execution error:" in observation
        )
        record = TurnRecord(
            turn_index=turn_idx,
            role="assistant",
            action=action,
            action_input=action_input,
            observation=observation,
            is_terminal=is_terminal,
            latency_ms=latency_ms,
            had_error=had_error,
        )
        self.turns.append(record)
        return record

    @property
    def current_turn(self) -> int:
        """Current number of executed turns."""
        return len(self.turns)

    @property
    def has_intermediate_error(self) -> bool:
        """Check if any non-terminal turn encountered an error."""
        return any(t.had_error and not t.is_terminal for t in self.turns)

    @property
    def is_finished(self) -> bool:
        """Check if episode reached terminal state or exceeded max turns."""
        if any(t.is_terminal for t in self.turns):
            return True
        return self.current_turn >= self.max_turns

    @property
    def final_query(self) -> str | None:
        """Get the final SQL query submitted by the agent, if any."""
        for t in reversed(self.turns):
            if t.action == "final_answer" and t.action_input:
                return str(t.action_input)
        return None

    def to_chat_history(self) -> list[dict[str, str]]:
        """Format trajectory into ChatML-compatible message list."""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a database agent. Use tools to inspect schemas and sample rows before answering.\n"
                    "Available tools:\n"
                    '- <tool name="inspect_schema">["table_name"]</tool>\n'
                    '- <tool name="sample_data">table_name</tool>\n'
                    '- <tool name="execute_sql">SELECT ...</tool>\n'
                    "- <final_answer>SELECT ...</final_answer>"
                ),
            },
            {
                "role": "user",
                "content": f"Database: {self.db_id}\nQuestion: {self.question}",
            },
        ]
        for t in self.turns:
            if t.action == "final_answer":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"<final_answer>{t.action_input}</final_answer>",
                    }
                )
            elif t.action:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f'<tool name="{t.action}">{t.action_input}</tool>',
                    }
                )
            else:
                messages.append({"role": "assistant", "content": str(t.action_input or "")})

            if not t.is_terminal:
                messages.append({"role": "tool", "content": t.observation})
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Convert interaction state to dictionary."""
        return {
            "question": self.question,
            "db_id": self.db_id,
            "db_path": self.db_path,
            "gold_sql": self.gold_sql,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "is_finished": self.is_finished,
            "has_intermediate_error": self.has_intermediate_error,
            "final_query": self.final_query,
            "turns": [t.to_dict() for t in self.turns],
        }
