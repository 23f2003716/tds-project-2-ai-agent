# Data Analyst Agent

An AI-powered data analysis tool that can process, analyze, and visualize data from various sources.

## Setup Instructions (Locally)

### Prerequisites

- Docker and Docker-Compose

### Environment Variables

Create a `.env` file in the root directory following `.env.example`:
#### To use multiple api keys in production and avoid rate limits

```
GEMINI_API_KEY_1=<your-gemini-api-key-1>
GEMINI_API_KEY_2=<your-gemini-api-key-2>
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

The API will be available at http://localhost:8000

### Testing

```bash
curl "http://0.0.0.0:8000/api/" -F "questions.txt=@questions.txt" -F "data.csv=@data.csv"
```

## API Endpoints

- `POST /api/` - Main analysis endpoint
- `GET /` - Root endpoint with health check

## Features

- Data processing and analysis
- Statistical analysis
- Data visualization
- Machine learning
- Web scraping
- Time series analysis
