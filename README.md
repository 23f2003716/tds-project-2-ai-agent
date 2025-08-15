# Data Analyst Agent

A powerful AI-driven data analysis platform that combines the capabilities of Google's Generative AI with advanced data processing tools to provide intelligent insights, visualizations, and automated analysis workflows.

## Setup Instructions (Locally)

### Prerequisites

- Docker and Docker-Compose

### Environment Variables

Create a `.env` file in the root directory following `.env.example`:

```
GEMINI_API_KEY=<your-gemini-api-key>
```
---

### Installation

```bash
docker-compose build data-analyst-agent
```

### Running the Application

```bash
docker-compose up -d
```

### Stopping the Application

```bash
docker-compose down
```

The API will be available at http://localhost:7860

### Testing

```bash
curl "http://0.0.0.0:7860/api/" -F "questions.txt=@questions.txt" -F "data.csv=@data.csv"
```

## API Endpoints

- `POST /api/` - Main analysis endpoint
- `GET /` - Root endpoint with health check

## Setup Instructions (Deploy at Huggingface)

### Follow any tutorial on how to create Huggingface Spaces

### Create a FREE Huggingface Space with blank Docker option

### Use this Dockerfile for Huggingface Spaces
``` Dockerfile
FROM astral/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl gcc g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set HOME and disable UV caching to avoid permission issues
ENV HOME=/app
ENV UV_NO_CACHE=1

RUN mkdir -p /app/.crawl4ai && chmod 777 /app/.crawl4ai
RUN mkdir -p /app/logs && chmod 777 /app/logs
RUN mkdir -p /app/uploads && chmod 777 /app/uploads
RUN mkdir -p /app/temp && chmod 777 /app/temp
RUN chmod 777 /app/

COPY pyproject.toml ./

RUN uv sync --no-cache

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

RUN mkdir -p /tmp/.chromium && chmod 777 /tmp/.chromium
ENV XDG_CONFIG_HOME=/tmp/.chromium
ENV XDG_CACHE_HOME=/tmp/.chromium

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

# Set up playwright for Crawl4AI with --no-cache
RUN uv run --no-cache playwright install --with-deps chromium

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

```

### Go to Settings of Spaces and add `GEMINI_API_KEY` to **Secrets**

## Features

- **🔍 Intelligent Data Analysis**: AI-powered insights using Google's Generative AI
- **📈 Interactive Visualizations**: Dynamic charts and graphs using Matplotlib and Seaborn
- **🌐 Web Scraping**: Extract data from URLs and web pages
- **📁 Multi-Format Support**: CSV, Excel, JSON, Parquet, and text files
- **🔄 Batch Processing**: Analyze multiple questions simultaneously
- **⚡ Real-time Processing**: Fast analysis with progress tracking
