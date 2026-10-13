# ------------------------------------------------------------
# AskTheDatabase Hugging Face Space & Container Deployment
# Multi-turn Text-to-SQL Agent and Interactive Streamlit UI
# ------------------------------------------------------------

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/chrisjcc/ask-the-database"
LABEL org.opencontainers.image.description="AskTheDatabase: Multi-turn reinforcement learning with verifiable rewards for text-to-SQL translation."

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY app/requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Application source
# ------------------------------------------------------------

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY pyproject.toml ./

# ------------------------------------------------------------
# Install AskTheDatabase package
# ------------------------------------------------------------

RUN pip install --no-cache-dir -e .

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8501

# ------------------------------------------------------------
# Start Streamlit
# ------------------------------------------------------------

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
