FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl gcc g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN uv sync

RUN export PATH="$(pwd)/.venv/bin:$PATH"

COPY . .

# Create necessary directories for the application
RUN mkdir -p logs temp

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set up Crawl4AI
RUN uv run crawl4ai-setup

EXPOSE 8000

CMD ["uv", "run", "main.py"]