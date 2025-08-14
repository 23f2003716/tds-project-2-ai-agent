import asyncio
import uvicorn
from agents import DataAnalystAgent
from config import logger
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Data Analyst Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DataAnalystAgent()


@app.post("/api/")
async def analyze_data(request: Request):
    """
    Main endpoint for data analysis, Accepts multiple files with any form field names
    """
    try:
        form = await request.form()

        if not form:
            raise HTTPException(status_code=400, detail="No files uploaded")

        files = []
        for _, value in form.items():
            if hasattr(value, 'filename') and value.filename:
                files.append(value)

        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        questions_file = None
        data_files = []

        for file in files:
            if file.filename and 'question' in file.filename.lower():
                questions_file = file
            else:
                data_files.append(file)

        if not questions_file:
            raise HTTPException(status_code=400, detail="questions.txt file is required")

        logger.info(f"Processing request for {questions_file.filename} with data files: {[f.filename for f in data_files]}")

        result = await asyncio.wait_for(
            agent.process_request(questions_file, data_files),
            timeout=240
        )

        return JSONResponse(content=result)

    except asyncio.TimeoutError:
        logger.error("Request timed out after 4 minutes")
        return JSONResponse(content={"error": "Request timed out after 4 mins"}, status_code=408)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
async def home():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
