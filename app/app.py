"""Interactive Streamlit application for multi-turn text-to-SQL exploration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from ask_the_database.environment.schema_manager import SchemaManager
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.inference.pipeline import NL2SQLPipeline

SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "concert_singer": [
        "Which singer has the highest average song release year?",
        "What are the names and capacities of all stadiums with capacity over 50,000?",
        "How many concerts were held in each year?",
    ],
    "car_1": [
        "Which country produces the highest number of car makers?",
        "What is the average horsepower of cars with 4 cylinders?",
        "List the names of all models made by Ford.",
    ],
    "pets_1": [
        "Find the pet type and weight of the heaviest pet.",
        "How many students have a pet?",
        "What is the average age of students who have a dog?",
    ],
    "flight_2": [
        "Which airline operates the most flights?",
        "What are the names of all airports located in California?",
        "Find the flight number with the longest distance.",
    ],
    "singer": [
        "Which singer has the highest net worth?",
        "List the songs and their sales for each singer.",
    ],
}


def init_demo_db(db_path: Path) -> None:
    """Initialize a realistic multi-table demo SQLite database (concert_singer) if none exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stadium (
                Stadium_ID INTEGER PRIMARY KEY,
                Location TEXT,
                Name TEXT,
                Capacity INTEGER,
                Highest INTEGER,
                Lowest INTEGER,
                Average INTEGER
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS singer (
                Singer_ID INTEGER PRIMARY KEY,
                Name TEXT,
                Country TEXT,
                Song_Name TEXT,
                Song_release_year INTEGER,
                Age INTEGER,
                Is_male TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS concert (
                concert_ID INTEGER PRIMARY KEY,
                concert_Name TEXT,
                Theme TEXT,
                Stadium_ID INTEGER,
                Year INTEGER,
                FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS singer_in_concert (
                concert_ID INTEGER,
                Singer_ID INTEGER,
                PRIMARY KEY (concert_ID, Singer_ID),
                FOREIGN KEY (concert_ID) REFERENCES concert(concert_ID),
                FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
            );
            """
        )
        cursor.execute("SELECT COUNT(*) FROM singer;")
        if cursor.fetchone()[0] == 0:
            singers = [
                (1, "Joe Sharp", "Netherlands", "You", 1992, 52, "F"),
                (2, "Timbaland", "United States", "Dangerous", 2008, 32, "T"),
                (3, "Justin Brown", "France", "Hey Oh", 2013, 29, "T"),
                (4, "Rose White", "France", "Sun", 2003, 41, "F"),
                (5, "John Doe", "United Kingdom", "Rain", 2011, 35, "T"),
            ]
            cursor.executemany("INSERT INTO singer VALUES (?, ?, ?, ?, ?, ?, ?);", singers)

            stadiums = [
                (1, "Raiders", "Stade de France", 80000, 80000, 75000, 78000),
                (2, "Giants", "MetLife Stadium", 82500, 82500, 80000, 81000),
                (3, "Wembley", "Wembley Stadium", 90000, 90000, 85000, 88000),
            ]
            cursor.executemany("INSERT INTO stadium VALUES (?, ?, ?, ?, ?, ?, ?);", stadiums)

            concerts = [
                (1, "Auditions", "Free Choice", 1, 2014),
                (2, "Super Bowl Live", "Bleed", 2, 2015),
                (3, "Summer Jam", "Joy", 3, 2016),
            ]
            cursor.executemany("INSERT INTO concert VALUES (?, ?, ?, ?, ?);", concerts)

            sic = [
                (1, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
            ]
            cursor.executemany("INSERT INTO singer_in_concert VALUES (?, ?);", sic)
        conn.commit()


@st.cache_resource
def get_pipeline(model_path: str, max_turns: int) -> NL2SQLPipeline:
    """Load and cache the multi-turn NL2SQL pipeline."""
    return NL2SQLPipeline(model_path=model_path, max_turns=max_turns)


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(page_title="AskTheDatabase 🤖", layout="wide")
    st.title("AskTheDatabase: Multi-Turn SQL Exploration Agent 🤖")

    # 1. Database Discovery across repository data and packaged demo databases
    db_map: dict[str, Path] = {}
    for base_dir in [Path("app/demo_databases"), Path("data/databases")]:
        if base_dir.exists():
            for d in sorted(base_dir.iterdir()):
                if d.is_dir():
                    sqlite_files = list(d.glob("*.sqlite"))
                    if sqlite_files and d.name not in db_map:
                        db_map[d.name] = sqlite_files[0]

    if not db_map:
        fallback_db = Path("app/demo_databases/concert_singer/concert_singer.sqlite")
        init_demo_db(fallback_db)
        db_map["concert_singer"] = fallback_db

    db_names = sorted(db_map.keys())
    default_idx = db_names.index("concert_singer") if "concert_singer" in db_names else 0

    # 2. Sidebar: Database & Model Configuration
    st.sidebar.header("Database Selection")
    selected_db = st.sidebar.selectbox("Choose a database:", db_names, index=default_idx)
    db_path = db_map[selected_db]

    st.sidebar.header("Model Configuration")
    default_model = (
        "models/rlvr/final" if Path("models/rlvr/final").exists() else "chrisjcc/ask-the-database"
    )
    model_path = st.sidebar.text_input("Checkpoint Path or HF Hub Repo:", value=default_model)
    max_turns = st.sidebar.slider("Max Interaction Turns:", min_value=1, max_value=10, value=5)

    pipeline = get_pipeline(model_path, max_turns)
    if pipeline.is_model_loaded:
        st.sidebar.success(f"🟢 **Neural Model Active**\n\nLoaded: `{model_path}`")
        if getattr(pipeline, "hardware_warning", None):
            st.sidebar.info(f"ℹ️ **Performance Notice:**\n\n{pipeline.hardware_warning}")
    else:
        st.sidebar.warning(
            "🟡 **Offline Mock Fallback Active**\n\n"
            "No neural weights loaded at the checkpoint path. "
            "Agent is running in deterministic heuristic exploration mode."
        )
        if pipeline.load_error:
            with st.sidebar.expander("Model Loading Diagnostics", expanded=False):
                st.caption(f"**Error Details:**\n```\n{pipeline.load_error}\n```")

    # 3. Main Header & Schema Transparency Drawer
    st.write(
        "Ask a complex natural language question about the database. "
        "The agent iteratively inspects schemas, samples rows, and executes verified queries."
    )

    sm = SchemaManager(str(db_path))
    tables = sm.get_tables()

    tables_label = ", ".join(f"`{t}`" for t in tables) if tables else "None"
    with st.expander(
        f"🗄️ **Active Database: `{selected_db}`** ({len(tables)} tables: {tables_label}) — View Schema DDL & Records",
        expanded=False,
    ):
        tab_ddl, tab_sample = st.tabs(["📋 Schema DDL", "🔍 Sample Records Preview"])
        with tab_ddl:
            st.code(sm.get_all_tables_ddl(), language="sql")
        with tab_sample:
            if tables:
                preview_table = st.selectbox(
                    "Select table to preview:", tables, key="preview_table_sel"
                )
                env_preview = SQLiteEnvironment(str(db_path))
                preview_res = env_preview.execute(f"SELECT * FROM `{preview_table}` LIMIT 5;")
                if preview_res.success and preview_res.columns:
                    st.dataframe(
                        pd.DataFrame(preview_res.data, columns=preview_res.columns),
                        width="stretch",
                    )
                else:
                    st.caption("No records available in this table.")
            else:
                st.caption("No tables found in this database.")

    # 4. Chat history initialization
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
                            language=("sql" if "SELECT" in str(step["output"]).upper() else "text"),
                        )
            if msg.get("query"):
                st.markdown(f"### Final SQL Query:\n```sql\n{msg['query']}\n```")
            if msg.get("data") is not None and msg.get("columns"):
                df = pd.DataFrame(msg["data"], columns=msg["columns"])
                st.dataframe(df, width="stretch")

    # 5. Suggested Example Questions
    examples = SAMPLE_QUESTIONS.get(
        selected_db, [f"SELECT * FROM `{tables[0]}` LIMIT 5;" if tables else "SELECT 1;"]
    )
    if examples:
        st.markdown("##### 💡 Suggested Questions")
        cols = st.columns(min(len(examples), 3))
        for idx, ex_q in enumerate(examples[:3]):
            if cols[idx].button(ex_q, key=f"btn_ex_{selected_db}_{idx}"):
                st.session_state.pending_prompt = ex_q
                st.rerun()

    # 6. User Prompt Acquisition
    prompt = None
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")
    else:
        prompt = st.chat_input("Ask a natural language question about the database...")

    if prompt:
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

            if not pipeline.is_model_loaded:
                st.caption(
                    "⚠️ *Executed via deterministic heuristic fallback (no neural weights loaded at checkpoint path).* "
                    "For neural text-to-SQL inference, load a fine-tuned model checkpoint."
                )

            st.markdown(f"### Final SQL Query:\n```sql\n{traj.final_query}\n```")

            result_data = None
            result_columns = None
            if traj.final_execution_result:
                if traj.final_execution_result.success:
                    result_data = traj.final_execution_result.data
                    result_columns = traj.final_execution_result.columns
                    if result_columns:
                        df = pd.DataFrame(result_data, columns=result_columns)
                        st.dataframe(df, width="stretch")
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
