import csv
import io
import json
import os
import re
import subprocess
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from datetime import datetime
from fastapi import UploadFile
from google.genai import types
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import client, logger


class DataAnalystAgent:
    def __init__(self):
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    async def extract_urls_from_questions(self, questions_content: str) -> List[str]:
        """Extract URLs from questions.txt content"""
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        urls = url_pattern.findall(questions_content)
        logger.info(f"Found {len(urls)} URLs in questions: {urls}")
        return urls

    async def scrape_url_content(self, url: str) -> str:
        """Scrape content from URL using crawl4ai in markdown format"""
        try:
            logger.info(f"Scraping URL: {url}")

            browser_config = BrowserConfig(
                headless=True,
                extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox",],
                viewport={"width": 1280, "height": 800},
                user_agent_mode="random",
                text_mode=True,
                light_mode=True,
                verbose=False)

            md_generator = DefaultMarkdownGenerator(
                content_source="cleaned_html",
                options={"ignore_links": True, "ignore_images": True}
            )

            crawl_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=md_generator,
                only_text=True,
                excluded_tags=["nav", "footer", "script", "style"]
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config
                )

                if result.success:
                    content = result.markdown
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

                    IMPORTANT FORMATTING RULES:
                    - Remove unnecessary text, markdown formatting, and HTML tags
                    - Keep data in a structured format (CSV preferred, then JSON, then markdown tables)
                    - Focus only on content needed to answer the questions
                    - Do NOT include code block markers like ```csv, ```json, ```markdown etc.
                    - Do NOT solve the questions yet, just extract the relevant data
                    - For CSV format: Use proper comma separation with headers in first row
                    - For JSON format: Use proper JSON structure without extra text
                    - For tables: Use clean pipe-separated format or CSV

                    Return ONLY the clean, structured data without any code block markers or explanatory text:

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
            return raw_content

    def clean_content_markers(self, content: str) -> str:
        """Remove code block markers and other formatting artifacts"""
        content = re.sub(r'```\w*\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple empty lines to double
        content = content.strip()

        return content

    def detect_content_type(self, content: str) -> Tuple[str, str]:
        """Detect content type and return (format, extension)"""
        cleaned_content = self.clean_content_markers(content)

        try:
            json.loads(cleaned_content)
            return 'json'
        except (json.JSONDecodeError, ValueError):
            pass

        if self.is_csv_content(cleaned_content):
            return 'csv'

        if self.is_markdown_content(cleaned_content):
            return 'md'

        return 'txt'

    def is_csv_content(self, content: str) -> bool:
        """Check if content appears to be CSV format"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check if first few lines have consistent comma/tab separation
        separators = [',', '\t',]
        for sep in separators:
            try:
                # Try to parse first few lines as CSV
                sample = '\n'.join(lines[:3])
                reader = csv.reader(io.StringIO(sample), delimiter=sep)
                rows = list(reader)

                if len(rows) >= 2:
                    # Check if rows have consistent column counts
                    col_counts = [len(row) for row in rows]
                    if len(set(col_counts)) == 1 and col_counts[0] > 1:
                        return True
            except:
                continue

        return False

    def is_markdown_content(self, content: str) -> bool:
        """Check if content appears to be markdown"""
        md_patterns = [
            r'^#{1,6}\s',  # Headers
            r'\*\*.*\*\*',  # Bold
            r'\*.*\*',  # Italic
            r'`.*`',  # Code
            r'\[.*\]\(.*\)',  # Links
            r'^\|.*\|',  # Tables
            r'^\s*[-*+]\s',  # Lists
            r'^\s*\d+\.\s',  # Numbered lists
        ]

        for pattern in md_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True

        return False

    async def save_scraped_file(self, url: str, content: str) -> str:
        """Save scraped content to file with proper extension and formatting"""
        cleaned_content = self.clean_content_markers(content)
        extension = self.detect_content_type(cleaned_content)

        safe_name = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))
        filename = f"scraped_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
        filepath = self.uploads_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            logger.info(f"Saved {extension} content to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving scraped file: {str(e)}")
            raise

    async def execute_python_code(self, code: str, file_paths: List[str]) -> Dict[str, Any]:
        """Execute Python code locally using uv and return the output"""
        try:

            cleaned_code = self.clean_content_markers(code)

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
                timeout=60
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
            uploaded_files = []
            file_names = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    base_name = os.path.basename(file_path)
                    if base_name.lower().startswith("questions") and "questions" in base_name.lower():
                        continue
                    try:
                        uploaded_file = client.files.upload(file=file_path)
                        uploaded_files.append(uploaded_file)
                        file_names.append(base_name)
                        logger.info(f"Uploaded file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error uploading {file_path}: {str(e)}")

            logger.info(f"Analyzing {len(file_names)} files with Gemini 2.5 Pro")

            prompt = f"""
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
                        *   Your code must print only the actual values of final answer(s), without any filler words or sentence.
                        *   If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output.

                    4.  **Output Serialization Mandate:**
                        *   **This is a non-negotiable requirement.** The final object printed to standard output MUST be a JSON-serializable string.
                        *   Use the `json.dumps()` function to create the final output string.
                        *   Any data structures containing NumPy types (e.g., `np.int64`, `np.float64`, `np.ndarray`) MUST be converted to their native Python equivalents (`int`, `float`, `list`) before being passed to `json.dumps()`.
                        *   For `np.ndarray`, use the `.tolist()` method.
                        *   For NumPy numeric types like `np.float64` or `np.int64`, cast them using `float()` or `int()`.
                        *   Failure to adhere to this will render the output unusable.

                    5.  Run the code to check if any errors are occuring then fix them accordingly.

                    6.  **Final Output:** Your response MUST contain ONLY the raw Python code that is running without errors. Do not include any explanations, comments, or markdown formatting like ```python... ```. Just the code itself.
                    
                    QUESTIONS TO ANSWER:
                    {questions_content}

                    EXACT FILE NAMES:
                    {file_names}
                    """

            response = await client.aio.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt, uploaded_files],
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
