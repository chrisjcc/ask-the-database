"""Shared pytest fixtures for ask-the-database."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def sample_sqlite_db(tmp_path: Path) -> Path:
    """Create a hermetic SQLite database with users, departments, and orders."""
    db_file = tmp_path / "company.sqlite"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE departments (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT NOT NULL,
            budget REAL
        );

        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            department_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES departments(department_id)
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        INSERT INTO departments VALUES (1, 'Engineering', 500000.0);
        INSERT INTO departments VALUES (2, 'Marketing', 200000.0);
        INSERT INTO departments VALUES (3, 'Finance', 300000.0);

        INSERT INTO users VALUES (1, 'Alice', 25, 1);
        INSERT INTO users VALUES (2, 'Bob', 30, 1);
        INSERT INTO users VALUES (3, 'Charlie', 20, 2);
        INSERT INTO users VALUES (4, 'Diana', 35, 3);
        INSERT INTO users VALUES (5, 'Evan', 40, NULL);

        INSERT INTO orders VALUES (101, 1, 150.0);
        INSERT INTO orders VALUES (102, 1, 250.0);
        INSERT INTO orders VALUES (103, 2, 90.0);
        INSERT INTO orders VALUES (104, 3, 300.0);
        """
    )
    conn.commit()
    conn.close()
    return db_file
