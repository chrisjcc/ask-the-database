"""Interactive Streamlit application for multi-turn text-to-SQL exploration."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from ask_the_database.environment.schema_manager import SchemaManager
from ask_the_database.inference.pipeline import NL2SQLPipeline


def init_demo_db(db_path: Path) -> None:
    """Initialize a lightweight fallback demo SQLite database if none exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS singer (
                Singer_ID INTEGER PRIMARY KEY,
                Name TEXT,
                Country TEXT,
                Song_Name TEXT,
                Song_release_year INTEGER,
                Age INTEGER
            );
            """
        )
        cursor.execute("SELECT COUNT(*) FROM singer;")
        if cursor.fetchone()[0] == 0:
            sample_data = [
                (1, "Joe Cocker", "Netherlands", "With a Little Help", 1969, 70),
                (2, "Adele", "UK", "Rolling in the Deep", 2010, 35),
                (3, "Ed Sheeran", "UK", "Shape of You", 2017, 33),
                (4, "Beyonce", "USA", "Halo", 2008, 42),
            ]
            cursor.executemany(
                "INSERT INTO singer VALUES (?, ?, ?, ?, ?, ?);",
                sample_data,
            )
        conn.commit()


@st.cache_resource
def get_pipeline(model_path: str, max_turns: int) -> NL2SQLPipeline:
    """Load and cache the multi-turn NL2SQL pipeline."""
    return NL2SQLPipeline(model_path=model_path, max_turns=max_turns)


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(page_title="AskTheDatabase 🤖", layout="wide")
    st.title("AskTheDatabase: Multi-Turn SQL Exploration Agent")

    # Sidebar: Database & Model Configuration
    st.sidebar.header("Database Selection")
    data_dir = Path("data/databases")
    db_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()]) if data_dir.exists() else []
    db_names = [d.name for d in db_dirs] if db_dirs else ["demo_db"]

    default_idx = db_names.index("singer") if "singer" in db_names else 0
    selected_db = st.sidebar.selectbox("Choose a database:", db_names, index=default_idx)

    db_path = Path(f"data/databases/{selected_db}/{selected_db}.sqlite")
    if not db_path.exists() and selected_db == "demo_db":
        init_demo_db(db_path)

    if db_path.exists():
        sm = SchemaManager(str(db_path))
        with st.sidebar.expander("View Schema DDL", expanded=False):
            st.code(sm.get_all_tables_ddl(), language="sql")
    else:
        st.sidebar.warning(f"Database file not found at: {db_path}")

    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.text_input("Checkpoint Path:", value="models/rlvr/final")
    max_turns = st.sidebar.slider("Max Interaction Turns:", min_value=1, max_value=10, value=5)

    st.write(
        "Ask a complex natural language question about the database. "
        "The agent iteratively inspects schemas, samples rows, and executes verified queries."
    )

    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display historical chat turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("Agent Thought Process & Exploration Steps", expanded=False):
                    for step in msg["steps"]:
                        st.write(f"**Turn {step['turn']}:** `{step['action']}`")
                        st.code(
                            step["output"],
                            language="sql" if "SELECT" in str(step["output"]).upper() else "text",
                        )
            if msg.get("query"):
                st.markdown(f"### Final SQL Query:\n```sql\n{msg['query']}\n```")
            if msg.get("data") is not None and msg.get("columns"):
                df = pd.DataFrame(msg["data"], columns=msg["columns"])
                st.dataframe(df, use_container_width=True)

    # User chat input
    placeholder = "e.g., Which singer has the highest average song release year?"
    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        pipeline = get_pipeline(model_path, max_turns)
        with st.chat_message("assistant"):
            with st.status("Agent exploring database...", expanded=True) as status:
                traj = pipeline.run_interactive(
                    question=prompt,
                    db_path=str(db_path),
                    db_id=selected_db,
                    max_turns=max_turns,
                )
                serialized_steps = []
                for step in traj.steps:
                    st.write(f"**Turn {step.turn}:** `{step.action}`")
                    is_sql = "SELECT" in str(step.output).upper()
                    st.code(step.output, language="sql" if is_sql else "text")
                    serialized_steps.append(
                        {
                            "turn": step.turn,
                            "action": step.action,
                            "action_input": step.action_input,
                            "output": step.output,
                        }
                    )
                status.update(label="Exploration complete!", state="complete")

            st.markdown(f"### Final SQL Query:\n```sql\n{traj.final_query}\n```")

            result_data = None
            result_columns = None
            if traj.final_execution_result:
                if traj.final_execution_result.success:
                    result_data = traj.final_execution_result.data
                    result_columns = traj.final_execution_result.columns
                    if result_columns:
                        df = pd.DataFrame(result_data, columns=result_columns)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Query executed successfully with no returned rows.")
                else:
                    st.error(f"Execution Error: {traj.final_execution_result.error_message}")

            assistant_msg = {
                "role": "assistant",
                "content": f"Resolved query for: **{prompt}**",
                "steps": serialized_steps,
                "query": traj.final_query,
                "data": result_data,
                "columns": result_columns,
            }
            st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    main()
