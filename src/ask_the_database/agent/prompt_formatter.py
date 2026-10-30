"""Prompt construction and action parsing for multi-turn SQL agents."""

import re

DEFAULT_SYSTEM_PROMPT = (
    "You are a database agent. Use tools to inspect schemas and sample rows before answering.\n"
    "Available tools:\n"
    '- <tool name="inspect_schema">["table_name"]</tool>\n'
    '- <tool name="sample_data">table_name</tool>\n'
    '- <tool name="execute_sql">SELECT ...</tool>\n'
    "- <final_answer>SELECT ...</final_answer>"
)


def format_chatml_prompt(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Format messages into ChatML tokens: <|im_start|>role\ncontent<|im_end|>."""
    formatted_parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        formatted_parts.append("<|im_start|>assistant\n")
    return "\n".join(formatted_parts)


def _clean_sql(query: str) -> str:
    """Clean markdown code blocks and trailing whitespace from SQL queries."""
    trimmed = query.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", trimmed, re.DOTALL | re.IGNORECASE)
    if match:
        trimmed = match.group(1).strip()
    return trimmed


def parse_agent_action(text: str) -> tuple[str, str]:
    """Extract tool call or final answer from agent completion text.

    Returns:
        tuple[str, str]: (action_name, action_payload)
    """
    # 1. Check for FinalAnswer tag
    final_match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return "final_answer", _clean_sql(final_match.group(1))

    # 2. Check for tool call tag
    tool_match = re.search(
        r'<tool\s+name=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</tool>',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if tool_match:
        name, payload = tool_match.group(1).strip(), tool_match.group(2).strip()
        if name == "execute_sql":
            payload = _clean_sql(payload)
        return name, payload

    # 3. Fallback: treat raw text as final answer attempt
    return "final_answer", _clean_sql(text.strip())
