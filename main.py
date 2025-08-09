import os
import re
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from random import randint
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from google import genai
from google.genai import types
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / f'api_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Initialize Gemini client
GEMINI_API_KEY = os.getenv(f"GEMINI_API_KEY_{randint(0, 1)}")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Data Analyst Agent", version="1.0.0")


class DataAnalystAgent:
    def __init__(self):
        self.uploads_dir = uploads_dir
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        
    async def extract_urls_from_questions(self, questions_content: str) -> List[str]:
        """Extract URLs from questions.txt content"""
        urls = self.url_pattern.findall(questions_content)
        logger.info(f"Found {len(urls)} URLs in questions: {urls}")
        return urls
    
    async def scrape_url_content(self, url: str) -> str:
        """Scrape content from URL using crawl4ai"""
        try:
            logger.info(f"Scraping URL: {url}")
            
            browser_config = BrowserConfig(headless=True, verbose=False)
            
            crawl_config = CrawlerRunConfig(
                word_count_threshold=15,
                extraction_strategy="NoExtractionStrategy",
                chunking_strategy="RegexChunking",
                cache_mode=CacheMode.BYPASS
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config
                )
                
                if result.success:
                    content = result.markdown or result.cleaned_html or result.html
                    logger.info(f"Successfully scraped {len(content)} characters from {url}")
                    return content
                else:
                    logger.error(f"Failed to scrape {url}: {result.error_message}")
                    return f"Error scraping {url}: {result.error_message}"
                    
        except Exception as e:
            logger.error(f"Exception while scraping {url}: {str(e)}")
            return f"Error scraping {url}: {str(e)}"
    
    async def process_scraped_content(self, url: str, raw_content: str, questions_content: str) -> str:
        """Use Gemini 2.5 Flash to extract relevant content from scraped data"""
        try:
            prompt = f"""
                    You are a data extraction specialist. I have scraped content from {url} and need to answer specific questions.

                    QUESTIONS TO ANSWER:
                    {questions_content}

                    Your task: Extract and clean only the data that is relevant to answering the questions above. 
                    - Remove unnecessary HTML, navigation, ads, etc.
                    - Keep data in a structured format (tables, lists, etc.)
                    - Focus only on content needed to answer the questions
                    - Do not solve the questions yet, just extract the relevant data
                    - Return the cleaned data in a format suitable for analysis

                    Provide clean, structured data only:

                    SCRAPED CONTENT:
                    {raw_content}
                    """
            
            logger.info(f"Processing scraped content with Gemini 2.5 Flash for {url}")
            
            response = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            
            cleaned_content = response.text
            logger.info(f"Processed content: {len(cleaned_content)} characters")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Error processing scraped content: {str(e)}")
            return raw_content  # Fallback to raw content
    
    async def save_scraped_file(self, url: str, content: str) -> str:
        """Save scraped content to file"""
        # Create safe filename from URL
        safe_name = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))
        filename = f"scraped_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = self.uploads_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n\n{content}")
            logger.info(f"Saved scraped content to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving scraped file: {str(e)}")
            raise
    
    async def analyze_with_gemini(self, questions_content: str, file_paths: List[str]) -> str:
        """Analyze all files with Gemini 2.5 Pro"""
        try:
            logger.info(f"Analyzing {len(file_paths)} files with Gemini 2.5 Pro")
            
            # Upload files to Gemini
            uploaded_files = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        uploaded_file = client.files.upload(file=file_path)
                        uploaded_files.append(uploaded_file)
                        logger.info(f"Uploaded file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error uploading {file_path}: {str(e)}")
            
            # Prepare content for analysis
            contents = [questions_content]
            contents.extend(uploaded_files)

            prompt = """
                    You are a world-class data analyst AI. Your purpose is to write robust, production-quality Python code to solve a user's question based on the data files they provide. 
                    You must follow these instructions meticulously:\n
                    1.  **Analyze the Request:** Carefully read the user's question and examine the previews of all provided files (text, CSV, images, etc.) to understand the context and requirements fully.
                    2.  **Think Step-by-Step:** Before writing code, formulate a clear plan. If links and scraping is mentioned, scraped data is already uploaded to you. Do NOT scrape anything. Consider data loading, necessary cleaning (handling missing values, correcting data types, ensuring case consistency), analysis steps, and the final output format.
                    3.  **Write High-Quality Python Code:
                        - The code must be pure Python and executable. Assume standard libraries like `pandas`, `matplotlib`, `numpy`, and `base64` are available.
                        - Refer to files by their exact filenames as provided (e.g., `sample-sales.csv`). Do not invent or assume file paths, instead robustly implement finding the filepath by filename to avoid FileNotFoundError.
                        - **Crucial:** Perform data cleaning and preprocessing. Do not make assumptions about data quality. Check for and handle inconsistencies.
                        - Your code must print the final answer(s) to standard output. The output format must precisely match what the user requested.
                        - If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output (e.g., `print(f'data:image/png;base64,{base64_string}')`).
                    4.  **Final Output:** Your response MUST contain ONLY the raw Python code. Do not include any explanations, comments, or markdown formatting like ```python ... ```. Just the code itself.
                    """
            
            # Generate response
            response = await client.aio.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt, contents],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    # max_output_tokens=8000,
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                )
            )
            
            result = response.text
            logger.info(f"Analysis complete: {len(result)} characters")
            
            # Cleanup uploaded files
            for uploaded_file in uploaded_files:
                try:
                    client.files.delete(name=uploaded_file.name)
                except:
                    pass  # Ignore cleanup errors
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            raise
    
    async def process_request(self, questions_file: UploadFile, additional_files: List[UploadFile]) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # Save questions file and read content
            questions_path = self.uploads_dir / f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(questions_path, 'wb') as f:
                content = await questions_file.read()
                f.write(content)
            
            questions_content = content.decode('utf-8')
            logger.info(f"Saved questions file: {questions_path}")
            
            # Save additional files
            saved_files = [str(questions_path)]
            for file in additional_files:
                file_path = self.uploads_dir / f"{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(file_path, 'wb') as f:
                    f.write(await file.read())
                saved_files.append(str(file_path))
                logger.info(f"Saved additional file: {file_path}")
            
            # Check for URLs in questions and scrape if needed
            urls = await self.extract_urls_from_questions(questions_content)
            
            if urls:
                logger.info(f"Found URLs, starting scraping process")
                for url in urls:
                    try:
                        # Scrape raw content
                        raw_content = await self.scrape_url_content(url)
                        
                        # Process with Gemini 2.5 Flash
                        cleaned_content = await self.process_scraped_content(url, raw_content, questions_content)
                        
                        # Save processed content
                        scraped_file_path = await self.save_scraped_file(url, cleaned_content)
                        saved_files.append(scraped_file_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        continue
            
            # Analyze all files with Gemini 2.5 Pro
            result = await self.analyze_with_gemini(questions_content, saved_files)
            
            # Try to parse as JSON, fall back to text if not valid JSON
            try:
                json_result = json.loads(result)
                return {"status": "success", "result": json_result}
            except json.JSONDecodeError:
                return {"status": "success", "result": result}
                
        except Exception as e:
            logger.error(f"Error in process_request: {str(e)}")
            return {"status": "error", "error": str(e)}


# Initialize agent
agent = DataAnalystAgent()


@app.post("/api/")
async def analyze_data(
    questions_file: UploadFile = File(alias="questions.txt"),
    files: List[UploadFile] = File(default=[])):
    """
    Main endpoint for data analysis
    Expects questions.txt and optional additional files
    """
    try:

        if not questions_file:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
        
        logger.info(f"Processing request with {len(files)} additional files")
        
        # Process with timeout (3 minutes)
        result = await asyncio.wait_for(
            agent.process_request(questions_file, files),
            timeout=180
        )
        
        return JSONResponse(content=result)
        
    except asyncio.TimeoutError:
        logger.error("Request timed out after 3 minutes")
        return JSONResponse(
            content={"status": "error", "error": "Request timed out after 3 minutes"},
            status_code=408
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/")
async def home():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")