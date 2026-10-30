"""Dataset formatting and loading for SFT warmup and RL rollouts.

Formats natural language questions, schema DDLs, and target SQL queries into
standard ChatML conversation representations.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert natural language to SQL translation system.\n"
    "Given a database schema DDL and a user question, generate a syntactically correct SQLite query.\n"
    "Strictly follow these rules:\n"
    "1. Only produce read-only queries (SELECT or WITH ... SELECT).\n"
    "2. Wrap your final SQL query in a markdown code block: ```sql\\n<QUERY>\\n```.\n"
    "3. Adhere to the provided schema and do not hallucinate tables or columns."
)


def format_chatml_messages(
    question: str,
    schema_ddl: str,
    gold_sql: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Format schema, question, and optional target SQL into ChatML messages.

    Args:
        question: User prompt natural language question.
        schema_ddl: Database schema definition in SQL DDL format.
        gold_sql: Ground-truth target SQL query (optional for inference).
        system_prompt: High-level instructions for the model.

    Returns:
        List of message dictionaries with 'role' and 'content' keys.
    """
    sys_content = f"{system_prompt}\n\n### Database Schema:\n{schema_ddl.strip()}"
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": question.strip()},
    ]
    if gold_sql is not None:
        messages.append({"role": "assistant", "content": f"```sql\n{gold_sql.strip()}\n```"})
    return messages


def format_prompt_for_generation(
    question: str,
    schema_ddl: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tokenizer: Any | None = None,
) -> str:
    """Format conversation into a prompt string ready for autoregressive completion.

    Args:
        question: User question.
        schema_ddl: Database schema DDL.
        system_prompt: System instructions.
        tokenizer: Optional Hugging Face tokenizer with apply_chat_template support.

    Returns:
        Formatted prompt string ending with assistant turn indicator.
    """
    messages = format_chatml_messages(
        question=question,
        schema_ddl=schema_ddl,
        gold_sql=None,
        system_prompt=system_prompt,
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        except Exception as e:
            logger.warning("Tokenizer apply_chat_template failed: %s. Falling back.", e)

    # Standard ChatML fallback formatting
    chatml_lines: list[str] = []
    for msg in messages:
        chatml_lines.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    chatml_lines.append("<|im_start|>assistant\n")
    return "\n".join(chatml_lines)


def load_sft_dataset(
    data_path: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_samples: int | None = None,
) -> Dataset:
    """Load JSONL dataset and convert to Hugging Face Dataset with ChatML messages.

    Args:
        data_path: Path to train.jsonl or val.jsonl.
        system_prompt: System prompt for conversational format.
        max_samples: Optional limit on the number of samples to load.

    Returns:
        Hugging Face Dataset with 'messages', 'gold_sql', and metadata columns.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"SFT dataset file not found at {data_path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = item.get("question", "")
            gold_sql = item.get("gold_sql", "")
            db_path = item.get("db_path", "")
            instance_id = item.get("instance_id", "")
            db_id = item.get("db_id", "")

            if "messages" in item and isinstance(item["messages"], list):
                messages = item["messages"]
            else:
                schema_ddl = item.get("schema_ddl", "")
                messages = format_chatml_messages(
                    question=question,
                    schema_ddl=schema_ddl,
                    gold_sql=gold_sql,
                    system_prompt=system_prompt,
                )

            records.append(
                {
                    "instance_id": instance_id,
                    "db_id": db_id,
                    "db_path": db_path,
                    "question": question,
                    "gold_sql": gold_sql,
                    "messages": messages,
                }
            )

            if max_samples is not None and len(records) >= max_samples:
                break

    logger.info("Loaded %d SFT samples from %s", len(records), data_path)
    return Dataset.from_list(records)


def extract_sql_from_completion(text: Any) -> str:
    """Extract SQL query from markdown code block or fallback to raw text.

    Args:
        text: Raw model generation string or conversational message sequence.

    Returns:
        Extracted and stripped SQL query string.
    """
    import re

    raw_str = ""
    if isinstance(text, list):
        if text and isinstance(text[-1], dict) and "content" in text[-1]:
            raw_str = str(text[-1]["content"])
        else:
            raw_str = str(text)
    else:
        raw_str = str(text)

    match = re.search(r"```sql\s*(.*?)\s*```", raw_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try generic markdown block
    generic_match = re.search(r"```\s*(.*?)\s*```", raw_str, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return raw_str.strip()
