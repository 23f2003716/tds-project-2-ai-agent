import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.usage import UsageLimits
import logfire
from dotenv import load_dotenv

from agent_tools import (
    DataAnalystDeps,
    setup_web_scraping_tool,
    setup_code_generation_tool
)

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
warnings.filterwarnings("ignore")
# Initialize FastAPI app
app = FastAPI(title="Data Analyst Agent API", version="1.0.0")

# Google GenAI model setup
provider = GoogleProvider(api_key=os.getenv("GEMINI_API_KEY"))
google_model = GoogleModel(model_name="gemini-2.5-flash", provider=provider)

# Response models
class AnalysisResponse(BaseModel):
    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task status: success, error, or partial")
    results: Dict[str, Any] = Field(description="Analysis results")
    visualizations: List[str] = Field(default=[], description="Paths to generated visualizations")
    code_generated: Optional[str] = Field(default=None, description="Generated code for the analysis")
    data_files: List[str] = Field(default=[], description="Processed data file paths")
    error_message: Optional[str] = Field(default=None, description="Error description if failed")

# Initialize the Data Analyst Agent
data_analyst_agent = Agent(
    model=google_model,
    deps_type=DataAnalystDeps,
    output_type=AnalysisResponse,
    system_prompt="""You are an expert Data Analyst Agent capable of:
    1. Web scraping and data collection from any source
    2. Data preprocessing, cleaning, and transformation
    3. Statistical analysis and modeling
    4. Data visualization creation
    5. Dynamic code generation for custom analysis tasks
    
    You have access to powerful tools for web scraping (Crawl4AI), data processing (pandas, numpy), 
    visualization (matplotlib, plotly, seaborn), and can generate and execute Python code dynamically.
    
    Always provide comprehensive analysis with clear insights, appropriate visualizations, 
    and well-documented code. Handle errors gracefully and provide alternative approaches when needed.
    
    Your responses should be professional, accurate, and actionable for business decision-making.""",
    instrument=True,

)

# Register tools with the agent
setup_web_scraping_tool(data_analyst_agent)
setup_code_generation_tool(data_analyst_agent)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent API is running", "version": "1.0.0"}


@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Main endpoint for data analysis tasks.
    
    Expects files with any names, but looks for:
    - questions.txt (or any .txt file) containing analysis questions/tasks
    - Any other files (.csv, .xlsx, .json, etc.) as data files
    
    Args:
        files: List of uploaded files including questions.txt and optional data files
    
    Returns:
        AnalysisResponse: Complete analysis results with visualizations and code
    """
    try:
        # Create temporary directory for this analysis session
        temp_dir = Path(tempfile.mkdtemp(prefix="data_analyst_"))
        
        # Separate questions file from data files
        questions_content = ""
        uploaded_file_paths = []
        
        for file in files:
            if not file.filename:
                continue
                
            # Read file content
            content = await file.read()
            file_path = temp_dir / file.filename
            
            # Save file to disk
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Check if this is the questions file
            if (file.filename.lower().endswith('.txt') or 
                'question' in file.filename.lower()):
                # This is likely the questions file
                try:
                    questions_content = content.decode('utf-8').strip()
                except UnicodeDecodeError:
                    # If it fails to decode, treat as a regular data file
                    uploaded_file_paths.append(str(file_path))
            else:
                # This is a data file
                uploaded_file_paths.append(str(file_path))
        
        if not questions_content:
            raise HTTPException(
                status_code=400, 
                detail="No questions file found. Please upload a .txt file with analysis questions."
            )
        
        # Create dependencies object
        deps = DataAnalystDeps(
            temp_dir=str(temp_dir),
            uploaded_files=uploaded_file_paths,
            task_id=f"task_{id(temp_dir)}"
        )
        
        # Run the agent with the analysis task
        result = await data_analyst_agent.run(
            user_prompt=f"""
            Please analyze the following data analysis request:
            
            {questions_content}
            
            Available data files: {uploaded_file_paths if uploaded_file_paths else 'None - you may need to collect data from web sources'}
            
            Provide a comprehensive analysis including:
            1. Data collection (if needed)
            2. Data preprocessing and cleaning
            3. Exploratory data analysis
            4. Statistical analysis/modeling (if applicable)
            5. Visualizations to support insights
            6. Generated code for reproducibility
            7. Clear business insights and recommendations
            """,
            deps=deps,
            usage_limits=UsageLimits(request_limit=5)
        )
        
        return result.data
        
    except Exception as e:
        # Clean up temporary directory on error
        if 'temp_dir' in locals():
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/simple")
async def analyze_simple(
    questions_txt: UploadFile = File(alias="questions.txt"),
    data_files: List[UploadFile] = File(default=[])
):
    """
    Alternative endpoint with explicit parameter names for curl usage.
    
    Usage:
    curl "http://localhost:8000/api/simple" -F "questions.txt=@questions.txt" -F "data_files=@data.csv"
    
    Args:
        questions_txt: Text file containing analysis questions/tasks
        data_files: Optional list of data files to analyze
    
    Returns:
        AnalysisResponse: Complete analysis results with visualizations and code
    """
    try:
        # Create temporary directory for this analysis session
        temp_dir = Path(tempfile.mkdtemp(prefix="data_analyst_"))
        
        # Read questions from uploaded file
        questions_content = await questions_txt.read()
        task_description = questions_content.decode('utf-8').strip()
        
        if not task_description:
            raise HTTPException(status_code=400, detail="Questions file is empty")
        
        # Save uploaded files and get their paths
        uploaded_file_paths = []
        for file in data_files:
            if file.filename:
                file_path = temp_dir / file.filename
                with open(file_path, 'wb') as f:
                    content = await file.read()
                    f.write(content)
                uploaded_file_paths.append(str(file_path))
        
        # Create dependencies object
        deps = DataAnalystDeps(
            temp_dir=str(temp_dir),
            uploaded_files=uploaded_file_paths,
            task_id=f"task_{id(temp_dir)}"
        )
        
        # Run the agent with the analysis task
        result = await data_analyst_agent.run(
            user_prompt=f"""
            Please analyze the following data analysis request:
            
            {task_description}
            
            Available data files: {uploaded_file_paths if uploaded_file_paths else 'None - you may need to collect data from web sources'}
            
            Provide a comprehensive analysis including:
            1. Data collection (if needed)
            2. Data preprocessing and cleaning
            3. Exploratory data analysis
            4. Statistical analysis/modeling (if applicable)
            5. Visualizations to support insights
            6. Generated code for reproducibility
            7. Clear business insights and recommendations
            """,
            deps=deps,
            usage_limits=UsageLimits(request_limit=5)
        )
        
        return result.data
        
    except Exception as e:
        # Clean up temporary directory on error
        if 'temp_dir' in locals():
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/web-scrape", response_model=Dict[str, Any])
async def web_scrape_endpoint(
    url: str = Form(..., description="URL to scrape"),
    query: str = Form(default="", description="Specific query for targeted scraping"),
    format: str = Form(default="markdown", description="Output format: markdown, json, text")
):
    """
    Dedicated web scraping endpoint
    
    Args:
        url: The URL to scrape
        query: Optional query for targeted content extraction
        format: Output format preference
    
    Returns:
        Dict containing scraped data and metadata
    """
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="web_scraper_"))
        
        deps = DataAnalystDeps(
            temp_dir=str(temp_dir),
            uploaded_files=[],
            task_id=f"scrape_{id(temp_dir)}"
        )
        
        prompt = f"""
        Please scrape the following URL and extract relevant data:
        URL: {url}
        Query: {query if query else 'Extract all meaningful content'}
        Format: {format}
        
        Return the scraped data in a structured format with metadata about the extraction.
        """
        
        result = await data_analyst_agent.run(user_prompt=prompt, deps=deps, usage_limits=UsageLimits(request_limit=5))
        
        return {
            "status": "success",
            "data": result.data.results,
            "url": url,
            "query": query,
            "format": format
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web scraping failed: {str(e)}"
        )

@app.post("/api/generate-code")
async def generate_code_endpoint(
    task_description: str = Form(..., description="Description of the code to generate"),
    language: str = Form(default="python", description="Programming language"),
    include_example: bool = Form(default=True, description="Include example usage")
):
    """
    Code generation endpoint
    
    Args:
        task_description: What the code should accomplish
        language: Programming language (default: python)
        include_example: Whether to include example usage
    
    Returns:
        Dict containing generated code and explanation
    """
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="code_gen_"))
        
        deps = DataAnalystDeps(
            temp_dir=str(temp_dir),
            uploaded_files=[],
            task_id=f"codegen_{id(temp_dir)}"
        )
        
        prompt = f"""
        Generate {language} code for the following task:
        {task_description}
        
        Requirements:
        - Include proper error handling
        - Add clear documentation/comments
        - Follow best practices
        - {'Include example usage' if include_example else 'No example needed'}
        
        Return the code with explanation of how it works.
        """
        
        result = await data_analyst_agent.run(user_prompt=prompt, deps=deps, usage_limits=UsageLimits(request_limit=5))
        
        return {
            "status": "success",
            "code": result.data.code_generated,
            "explanation": result.data.results,
            "language": language,
            "task": task_description
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")