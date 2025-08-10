import json
import logging
import os
import re
import subprocess
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from datetime import datetime, timezone, timedelta
from fastapi import UploadFile
from google import genai
from google.genai import types
from pathlib import Path
from random import randint
from typing import List, Dict, Any

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

ist = timezone(timedelta(hours=5, minutes=30))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / f'api_{datetime.now(ist).strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Configure timezone for all log timestamps
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if hasattr(handler.formatter, 'converter'):
        handler.formatter.converter = lambda *args: datetime.now(ist).timetuple()

logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.getenv(f"GEMINI_API_KEY_{randint(0, 1)}")
client = genai.Client(api_key=GEMINI_API_KEY)


class DataAnalystAgent:
    def __init__(self):
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

    async def extract_urls_from_questions(self, questions_content: str) -> List[str]:
        """Extract URLs from questions.txt content"""
        urls = self.url_pattern.findall(questions_content)
        logger.info(f"Found {len(urls)} URLs in questions: {urls}")
        return urls

    async def scrape_url_content(self, url: str) -> str:
        """Scrape content from URL using crawl4ai in markdown format"""
        try:
            logger.info(f"Scraping URL: {url}")

            browser_config = BrowserConfig(headless=True, verbose=False)

            crawl_config = CrawlerRunConfig(
                word_count_threshold=15,
                cache_mode=CacheMode.BYPASS,
                markdown_generator="DEFAULT"
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config
                )

                if result.success:
                    # Prioritize markdown content
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

    def get_file_extension(self, url: str, content: str) -> str:
        """Determine appropriate file extension based on content"""
        try:
            json.loads(content)
            return 'json'
        except json.JSONDecodeError:
            pass

        # Check if content looks has markdown syntax
        if any(marker in content for marker in ['#', '**', '*', '`', '[', ']', '|']):
            return 'md'

        return 'txt'

    async def save_scraped_file(self, url: str, content: str) -> str:
        """Save scraped content to file with proper extension"""
        safe_name = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))

        extension = self.get_file_extension(url, content)
        filename = f"scraped_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
        filepath = self.uploads_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if extension == 'json':
                    try:
                        json_content = json.loads(content)
                        json.dump(json_content, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        f.write(f"URL: {url}\n\n{content}")
                else:
                    f.write(f"URL: {url}\n\n{content}")

            logger.info(f"Saved scraped content to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving scraped file: {str(e)}")
            raise

    def clean_code_block(self, code: str) -> str:
        """
        Remove markdown code block formatting if present using a regular expression.
        Handles optional language identifiers and surrounding newlines.
        """
        pattern = r"^```(?:\w*\s*)?\n?(.*?)\n?```$"
        cleaned_code = re.sub(pattern, r'\1', code.strip(), flags=re.DOTALL)

        return cleaned_code.strip()

    async def execute_python_code(self, code: str, file_paths: List[str]) -> Dict[str, Any]:
        """Execute Python code locally using uv and return the output"""
        try:

            cleaned_code = self.clean_code_block(code)

            for file_path in file_paths:
                if os.path.exists(file_path):
                    source = Path(file_path)
                    dest = self.temp_dir / source.name
                    dest.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')

            code_file = self.temp_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            code_file.write_text(cleaned_code, encoding='utf-8')

            result = subprocess.run(
                ["uv", "run", code_file.name],
                cwd=str(self.temp_dir),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                output = result.stdout.strip()

                try:
                    return {"success": True, "result": json.loads(output)}
                except json.JSONDecodeError:
                    return {"success": True, "result": output}
            else:
                logger.error(f"Code execution failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

        except subprocess.TimeoutExpired:
            logger.error("Code execution timed out")
            return {"success": False, "error": "Code execution timed out"}
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return {"success": False, "error": str(e)}

    async def analyze_with_gemini(self, questions_content: str, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze all files with Gemini 2.5 Pro and execute code locally"""
        try:
            logger.info(f"Analyzing {len(file_paths)} files with Gemini 2.5 Pro")

            uploaded_files = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        uploaded_file = client.files.upload(file=file_path)
                        uploaded_files.append(uploaded_file)
                        logger.info(f"Uploaded file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error uploading {file_path}: {str(e)}")

            contents = [questions_content]
            contents.extend(uploaded_files)

            prompt = """
                    You are a world-class data analyst AI who is also an expert Python developer with deep knowledge of data serialization standards for production systems. Your purpose is to write robust, production-quality Python code to solve a user's question based on the data files they provide. 

                    You must follow these instructions meticulously:

                    1.  **Analyze the Request:** Carefully read the user's question and examine the previews of all provided files (text, CSV, images, etc.) to understand the context and requirements fully.

                    2.  **Think Step-by-Step:** Before writing code, formulate a clear plan inside a `<thought>` block. If links and scraping are mentioned, assume the scraped data is already uploaded to you. Do NOT scrape anything. Consider data loading, necessary cleaning (handling missing values, correcting data types), analysis steps, and the final output format. Your thought process is for your own guidance and should not be in the final Python code.

                    3.  **Write High-Quality Python Code:**
                        *   The code must be pure Python and executable.
                        *   Include `uv` script dependencies at the top of the code, for example:
                            ```python
                            # /// script
                            # requires-python = ">=3.13"
                            # dependencies = ["pandas", "numpy"]
                            # ///
                            ```
                        *   Refer to files by their exact filenames. Do not assume file paths; instead, robustly implement finding the filepath by filename (e.g., using `os.walk`) to avoid `FileNotFoundError`.
                        *   Perform data cleaning and preprocessing. Do not make assumptions about data quality. Check for and handle inconsistencies.
                        *   Your code must print the final answer(s) to standard output.
                        *   If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output (e.g., `print(f'data:image/png;base64,{base64_string}')`).

                    4.  **Output Serialization Mandate:**
                        *   **This is a non-negotiable requirement.** The final object printed to standard output MUST be a JSON-serializable string.
                        *   Use the `json.dumps()` function to create the final output string.
                        *   Any data structures containing NumPy types (e.g., `np.int64`, `np.float64`, `np.ndarray`) MUST be converted to their native Python equivalents (`int`, `float`, `list`) before being passed to `json.dumps()`.
                        *   For `np.ndarray`, use the `.tolist()` method.
                        *   For NumPy numeric types like `np.float64` or `np.int64`, cast them using `float()` or `int()`.
                        *   Failure to adhere to this will render the output unusable.

                    5.  Run the code to check if any errors are occuring then fix them accordingly..


                    6.  **Final Output:** Your response MUST contain ONLY the raw Python code that is running without errors. Do not include any explanations, comments, or markdown formatting like ```python... ```. Just the code itself.
                    """

            response = await client.aio.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt, contents],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                )
            )

            generated_code = response.text.strip()
            logger.info(f"Generated code: {len(generated_code)} characters")

            execution_result = await self.execute_python_code(generated_code, file_paths)

            for uploaded_file in uploaded_files:
                try:
                    client.files.delete(name=uploaded_file.name)
                except:
                    pass

            return execution_result

        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            raise

    async def process_request(self, questions_file: UploadFile, additional_files: List[UploadFile]) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            questions_path = self.uploads_dir / f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(questions_path, 'wb') as f:
                content = await questions_file.read()
                f.write(content)

            questions_content = content.decode('utf-8')
            logger.info(f"Saved questions file: {questions_path}")

            saved_files = [str(questions_path)]
            for file in additional_files:
                file_path = self.uploads_dir / f"{file.filename}"
                with open(file_path, 'wb') as f:
                    f.write(await file.read())
                saved_files.append(str(file_path))
                logger.info(f"Saved additional file: {file_path}")

            urls = await self.extract_urls_from_questions(questions_content)

            if urls:
                logger.info(f"Found URLs, starting scraping process")
                for url in urls:
                    try:
                        raw_content = await self.scrape_url_content(url)

                        cleaned_content = await self.process_scraped_content(url, raw_content, questions_content)

                        scraped_file_path = await self.save_scraped_file(url, cleaned_content)
                        saved_files.append(scraped_file_path)

                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        continue

            result = await self.analyze_with_gemini(questions_content, saved_files)

            if result["success"]:
                return result["result"]
            else:
                return {"error": result["error"]}

        except Exception as e:
            logger.error(f"Error in process_request: {str(e)}")
            return {"error": str(e)}
