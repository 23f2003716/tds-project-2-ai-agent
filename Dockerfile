FROM astral/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl gcc g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN uv sync

RUN export PATH="$(pwd)/.venv/bin:$PATH"

COPY . .

# Create necessary directories for the application
RUN mkdir -p logs uploads temp

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

# Set up playwright for Crawl4AI
RUN uv run playwright install --with-deps chromium

EXPOSE 8000

CMD ["uv", "run", "main.py"]