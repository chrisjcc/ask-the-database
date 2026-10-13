# AskTheDatabase — Phase 4 Granular Implementation Plan
## Multi-Turn Agent Environment & SkyRL Integration (Issues #16 to #20)

---

## 1. Phase 4 Overview & Scientific Scope

The core objective of **Phase 4** is to implement the **Multi-Turn Agentic Database Exploration Architecture**. Single-turn text-to-SQL forces the model to perform schema linking, join deduction, syntax construction, and semantic validation in a single forward pass. When working with enterprise schemas (dozens of tables, hundreds of columns), single-turn models suffer from hallucinated column names, invalid joins, and context dilution.

Phase 4 bridges the **SkyRL-Agent** framework (*"SKYRL-SQL: Multi-turn SQL Data Agents via RL"*, NovaSky AI, 2025) into the repository:
1. **Interactive Markov Decision Process (MDP)**: Formulating text-to-SQL as an interactive loop where the model is an agent equipped with database inspection tools.
2. **Tool Registry**: Providing `inspect_schema`, `sample_data`, and `execute_sql` actions.
3. **SkyRL-Gym Environment**: Wrapping the SQLite sandbox into a Gymnasium-compatible environment for agent training.
4. **Step-Discount & Recovery Rewards**: Penalizing excessive exploration turns while explicitly rewarding the model for self-correcting after receiving database execution errors.
5. **Multi-Turn RLVR Training Pipeline**: Orchestrating agentic policy updates using SkyRL-Agent rollouts.

---

## 2. Directory Layout & Target File Inventory

```text
ask-the-database/
├── configs/
│   └── training/
│       └── rlvr_skyrl_agent.yaml     # Multi-turn SkyRL-Agent training config
│
├── src/
│   └── ask_the_database/
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── interaction_state.py   # Issue #16: Turn history & state tracking
│       │   ├── tool_registry.py       # Issue #16: DB probing tools
│       │   └── prompt_formatter.py    # Multi-turn ChatML prompt constructor
│       ├── rewards/
│       │   └── multi_turn_reward.py   # Issue #18: Step decay & recovery bonus
│       └── training/
│           └── adapters/
│               └── skyrl_adapter.py   # Issue #17: SkyRL-Gym environment bridge
│
├── scripts/
│   └── train_rlvr.py                  # Issue #19: Multi-turn RL training entrypoint
│
└── tests/
    └── integration/
        └── test_multiturn.py          # Issue #20: Full trajectory rollout test
```

---

## 3. Granular Issue-by-Issue Implementation Specifications

---

### Issue #16: Multi-Turn Interaction State & Tool Registry

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/agent/tool_registry.py` and `src/ask_the_database/agent/interaction_state.py`.
- Define the 4 discrete actions the agent can take:
  1. `Tool: inspect_schema(tables: list[str])`: Returns DDL and foreign keys for requested tables.
  2. `Tool: sample_data(table: str, limit: int = 3)`: Returns first 3 rows of a table to inspect casing, formats, and categories.
  3. `Tool: execute_sql(query: str)`: Runs a trial query in the read-only sandbox and returns the tabular result or raw error message.
  4. `Action: FinalAnswer(query: str)`: Submits the final query for formal reward evaluation and terminates the episode.
- Track trajectory history, tool inputs/outputs, and step latency.

#### 2. Exact File Contracts

##### `src/ask_the_database/agent/tool_registry.py`
```python
from dataclasses import dataclass
from typing import Any
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.environment.schema_manager import SchemaManager


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool invocation."""
    action: str
    output: str
    is_terminal: bool
    final_query: str | None = None


class DatabaseToolRegistry:
    """Registry of interactive tools exposed to the SQL agent."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.env = SQLiteEnvironment(db_path)
        self.schema_manager = SchemaManager(db_path)

    def inspect_schema(self, table_names: list[str]) -> ToolResult:
        """Inspect table DDL and relationships."""
        ddl_chunks = []
        for table in table_names:
            ddl = self.schema_manager.get_table_ddl(table)
            if ddl:
                ddl_chunks.append(ddl)
            else:
                ddl_chunks.append(f"Error: Table '{table}' not found in database.")
        return ToolResult(
            action="inspect_schema",
            output="\n\n".join(ddl_chunks),
            is_terminal=False,
        )

    def sample_data(self, table_name: str, limit: int = 3) -> ToolResult:
        """Sample rows from a specific table."""
        res = self.env.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        if not res.success:
            return ToolResult(
                action="sample_data",
                output=f"Execution error: {res.error_message}",
                is_terminal=False,
            )
        col_str = " | ".join(res.columns or [])
        row_strs = [" | ".join(str(cell) for cell in row) for row in (res.data or [])]
        table_output = f"{col_str}\n" + "-" * len(col_str) + "\n" + "\n".join(row_strs)
        return ToolResult(
            action="sample_data",
            output=table_output,
            is_terminal=False,
        )

    def execute_sql(self, sql_query: str) -> ToolResult:
        """Run trial query against database sandbox."""
        res = self.env.execute(sql_query)
        if not res.success:
            return ToolResult(
                action="execute_sql",
                output=f"Database Error: {res.error_message}",
                is_terminal=False,
            )
        row_preview = str(res.data[:5]) if res.data else "[]"
        return ToolResult(
            action="execute_sql",
            output=f"Success: {res.row_count} rows returned. Preview: {row_preview}",
            is_terminal=False,
        )

    def final_answer(self, sql_query: str) -> ToolResult:
        """Submit final SQL query and terminate episode."""
        return ToolResult(
            action="final_answer",
            output="Submitted final query.",
            is_terminal=True,
            final_query=sql_query,
        )
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_tool_registry.py`:
- `inspect_schema` returns valid DDL strings.
- `sample_data` returns formatted table rows without mutating state.
- `execute_sql` executes trial queries and returns previews.
- `final_answer` flags `is_terminal=True`.

---

### Issue #17: SkyRL-Gym Environment Implementation

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/training/adapters/skyrl_adapter.py`.
- Wrap the multi-turn database interaction loop into a Gymnasium-compatible environment (`gym.Env`) following SkyRL-Gym specifications.
- Define observation space (tokenized conversational history) and action space (text generations containing tool calls or final answer).
- Parse XML/JSON tool call tags from model completions:
  - `<tool name="inspect_schema">["employees"]</tool>`
  - `<final_answer>SELECT * FROM employees</final_answer>`

#### 2. Exact File Contracts

##### `src/ask_the_database/training/adapters/skyrl_adapter.py`
```python
from typing import Any
import re
from omegaconf import DictConfig
from ask_the_database.agent.tool_registry import DatabaseToolRegistry
from ask_the_database.rewards.multi_turn_reward import MultiTurnRewardEvaluator


class SkyRLSQLEnvironment:
    """Gymnasium-compatible multi-turn SQL environment for SkyRL-Agent."""

    def __init__(self, config: DictConfig) -> None:
        self.cfg = config
        self.max_turns = config.get("max_turns", 5)
        self.reward_evaluator = MultiTurnRewardEvaluator(config)
        self.current_turn = 0
        self.history: list[dict[str, str]] = []
        self.registry: DatabaseToolRegistry | None = None
        self.gold_sql: str = ""
        self.db_path: str = ""

    def reset(self, instance: dict[str, Any]) -> str:
        """Reset environment with a new benchmark question and database."""
        self.current_turn = 0
        self.gold_sql = instance["gold_sql"]
        self.db_path = instance["db_path"]
        self.registry = DatabaseToolRegistry(self.db_path)
        self.history = [
            {"role": "system", "content": "You are a database agent. Use tools to inspect schemas and sample rows before answering."},
            {"role": "user", "content": f"Database: {instance['db_id']}\nQuestion: {instance['question']}"},
        ]
        return self._format_observation()

    def _parse_action(self, text: str) -> tuple[str, Any]:
        """Extract tool call or final answer from LLM completion."""
        # Check for FinalAnswer
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL | re.IGNORECASE)
        if final_match:
            return "final_answer", final_match.group(1).strip()

        # Check for tool call
        tool_match = re.search(r'<tool\s+name="([^"]+)">(.*?)</tool>', text, re.DOTALL | re.IGNORECASE)
        if tool_match:
            name, payload = tool_match.group(1), tool_match.group(2).strip()
            return name, payload

        # Fallback: treat raw text as final answer attempt
        return "final_answer", text.strip()

    def step(self, model_completion: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Advance the environment by executing the model's action."""
        self.current_turn += 1
        action_name, action_payload = self._parse_action(model_completion)

        assert self.registry is not None
        if action_name == "inspect_schema":
            tables = [t.strip().strip('"\'') for t in action_payload.strip("[]").split(",")]
            tool_res = self.registry.inspect_schema(tables)
        elif action_name == "sample_data":
            tool_res = self.registry.sample_data(action_payload)
        elif action_name == "execute_sql":
            tool_res = self.registry.execute_sql(action_payload)
        else:  # final_answer
            tool_res = self.registry.final_answer(action_payload)

        is_done = tool_res.is_terminal or (self.current_turn >= self.max_turns)

        # Compute multi-turn reward
        reward = self.reward_evaluator.evaluate_step(
            action_name=action_name,
            tool_output=tool_res.output,
            final_query=tool_res.final_query,
            gold_sql=self.gold_sql,
            db_path=self.db_path,
            turn=self.current_turn,
            is_terminal=is_done,
        )

        self.history.append({"role": "assistant", "content": model_completion})
        self.history.append({"role": "tool", "content": tool_res.output})

        return self._format_observation(), reward, is_done, {"turn": self.current_turn}

    def _format_observation(self) -> str:
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_skyrl_adapter.py`:
- Agent completes 3 tool calls and 1 final answer.
- Environment advances step counter and terminates correctly on `<final_answer>`.

---

### Issue #18: Multi-Turn Step Discount & Self-Correction Rewards

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/rewards/multi_turn_reward.py`.
- Formulate the multi-turn reward dynamics:
  1. **Turn Budget Discount**: Apply a small penalty $-0.05 \times t$ per step to encourage the model to query efficiently and prevent infinite loops.
  2. **Self-Correction Recovery Bonus**: If the model ran a trial query at step $t-1$ that produced a syntax or database error, and at step $t$ successfully resolved the error and achieved execution equivalence, award $+0.3$ recovery bonus.
  3. **Terminal Reward**: The underlying `SQLRewardRubric` from Phase 2 is evaluated on the final query.

#### 2. Exact File Contracts

##### `src/ask_the_database/rewards/multi_turn_reward.py`
```python
from omegaconf import DictConfig
from ask_the_database.rewards.rubric import SQLRewardRubric


class MultiTurnRewardEvaluator:
    """Evaluates intermediate tool steps and terminal multi-turn trajectories."""

    def __init__(self, config: DictConfig) -> None:
        self.cfg = config
        self.rubric = SQLRewardRubric(config)
        self.step_penalty = config.get("step_penalty", -0.05)
        self.recovery_bonus = config.get("recovery_bonus", 0.3)
        self.had_intermediate_error = False

    def evaluate_step(
        self,
        action_name: str,
        tool_output: str,
        final_query: str | None,
        gold_sql: str,
        db_path: str,
        turn: int,
        is_terminal: bool,
    ) -> float:
        """Calculate step reward incorporating discount and self-correction bonus."""
        # Non-terminal intermediate tool step
        if not is_terminal:
            if "Database Error" in tool_output or "Error:" in tool_output:
                self.had_intermediate_error = True
            return self.step_penalty

        # Terminal step (Final Answer or max turns reached)
        if final_query is None:
            return -1.0  # Exhausted budget without submitting final answer

        base_res = self.rubric.evaluate(final_query, gold_sql, db_path)
        terminal_reward = base_res.total_reward

        # Apply turn discount
        discounted_reward = terminal_reward - (self.step_penalty * turn)

        # Apply recovery bonus if model fixed an earlier runtime error
        if base_res.is_execution_equivalent and self.had_intermediate_error:
            discounted_reward += self.recovery_bonus

        return max(-1.0, min(1.0, discounted_reward))
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_multiturn_rewards.py`:
- 1-turn perfect answer receives maximum reward ($1.0$).
- 3-turn answer with recovery from a syntax error receives discounted reward $+$ recovery bonus.

---

### Issue #19: Multi-Turn RLVR Training Pipeline

#### 1. Objectives & Responsibilities
- Implement `scripts/train_rlvr.py`.
- Initialize `SkyRLSQLEnvironment` using `configs/training/rlvr_skyrl_agent.yaml`.
- Load warmed-up SFT checkpoint from `models/sft/final`.
- Execute multi-turn agentic rollouts:
  - Policy generates tool calls, receives environment observations, updates context, and generates final query.
  - Compute group relative advantages across 8 rollouts per problem.
  - Optimize policy parameters using GRPO / PPO loss.
- Persist aligned checkpoint to `models/rlvr/final`.

#### 2. Exact File Contracts

##### Hydra Configuration: `configs/training/rlvr_skyrl_agent.yaml`
```yaml
model_path: "models/sft/final"
output_dir: "models/rlvr/final"
learning_rate: 1.0e-6
num_rollouts_per_prompt: 8
max_turns: 5
step_penalty: -0.05
recovery_bonus: 0.3
beta: 0.05
max_steps: 250
logging_steps: 5
save_steps: 50
report_to: "wandb"
```

#### 3. Definition of Done & Verification Gate
```bash
python scripts/train_rlvr.py training.max_steps=5
```
*Pass Criteria*: Multi-turn rollouts execute across 5 steps, tools execute within sandbox, and W&B logs multi-turn rewards.

---

### Issue #20: Multi-Turn Interactive Integration Tests

#### 1. Objectives & Responsibilities
- Implement `tests/integration/test_multiturn.py`.
- Simulate end-to-end agentic trajectory without mocking:
  - Agent starts with user query $\rightarrow$ calls `inspect_schema` $\rightarrow$ calls `sample_data` $\rightarrow$ calls `execute_sql` $\rightarrow$ submits `<final_answer>`.
  - Assert that database states remain completely unmutated.
  - Assert that intermediate tool outputs correctly feed into next-turn prompt context.

#### 2. Exact Test Specification
```python
def test_full_multiturn_agentic_trajectory(test_db):
    env = SkyRLSQLEnvironment(config)
    obs = env.reset({"question": "Who is the oldest singer?", "gold_sql": "SELECT name FROM singer ORDER BY age DESC LIMIT 1;", "db_id": "singer", "db_path": test_db})
    
    # Turn 1: Inspect
    obs, r1, done, _ = env.step('<tool name="inspect_schema">["singer"]</tool>')
    assert not done and "singer_id" in obs
    
    # Turn 2: Final Answer
    obs, r2, done, _ = env.step('<final_answer>SELECT name FROM singer ORDER BY age DESC LIMIT 1;</final_answer>')
    assert done and r2 > 0.8
```

#### 3. Definition of Done & Verification Gate
```bash
pytest tests/integration/test_multiturn.py -v
```
*Pass Criteria*: 100% passing across end-to-end multi-turn trajectories.

---

## 4. Phase 4 Summary & Transition to Phase 5

With Phase 4 complete:
- The model is trained as an autonomous, multi-turn database exploration agent.
- Intermediate tool calls, error self-correction, and terminal rewards are fully integrated.
- **Phase 5** will finalize the project by implementing the **Comprehensive Benchmarking Suite** (Spider, DK, Syn, Realistic), **Bayesian Hyperparameter Sweeps**, **Hardened Slurm Operations**, and the **Interactive Streamlit Demo**.
