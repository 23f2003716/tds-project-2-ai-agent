import asyncio
import csv
import io
import json
import os
import re
import subprocess
from config import client, logger
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from datetime import datetime
from fastapi import UploadFile
from google.genai import types
from pathlib import Path
from prompts import scraping_prompt, cleaning_prompt, analysis_prompt
from typing import List, Dict, Any, Optional


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
                extra_args=[
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-crashpad",
                    "--disable-crash-reporter",
                    "--disable-logging",
                    "--silent",
                ],
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
                result = await crawler.arun(url=url, config=crawl_config)

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

    async def check_scraping_requirement(self, questions_content: str) -> bool:
        """Use Gemini to check if data needs to be scraped"""
        try:
            prompt = scraping_prompt(questions_content)

            logger.info(f"Checking if data needs to be scraped with Gemini")

            response = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )

            if response and hasattr(response, 'text') and response.text:
                result = response.text.strip().lower()
                return result == "true"
            else:
                logger.warning("Empty response from Gemini Flash, defaulting to no scraping")
                return False

        except Exception as e:
            logger.error(f"Error checking scraping requirement: {str(e)}")
            return False

    async def process_scraped_content(self, url: str, raw_content: str, questions_content: str) -> str:
        """Use Gemini to extract relevant content from scraped data"""
        try:
            prompt = cleaning_prompt(url, raw_content, questions_content)

            logger.info(f"Processing scraped content with Gemini for {url}")

            response = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )

            cleaned_content = response.text
            logger.info(f"Processed content: {len(cleaned_content)} characters")
            return str(cleaned_content)

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

    def detect_content_type(self, content: str) -> str:
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

        separators = [',', '\t',]
        for sep in separators:
            try:
                sample = '\n'.join(lines[:3])
                reader = csv.reader(io.StringIO(sample), delimiter=sep)
                rows = list(reader)

                if len(rows) >= 2:
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

    async def scrape_urls_parallel(self, urls: List[str], questions_content: str, max_concurrent: int = 5) -> List[str]:
        """Scrape multiple URLs in parallel with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_single_url(url: str) -> Optional[str]:
            async with semaphore:
                try:
                    logger.info(f"Starting scrape for URL: {url}")
                    raw_content = await self.scrape_url_content(url)
                    cleaned_content = await self.process_scraped_content(url, raw_content, questions_content)
                    scraped_file_path = await self.save_scraped_file(url, cleaned_content)
                    logger.info(f"Successfully scraped and saved: {url}")
                    return scraped_file_path
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    return None

        tasks = [scrape_single_url(url) for url in urls]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_files = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception occurred for URL {urls[i]}: {str(result)}")
            elif result is not None:
                scraped_files.append(result)

        logger.info(f"Completed parallel scraping. Successfully processed {len(scraped_files)} out of {len(urls)} URLs")
        return scraped_files

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
                ["uv", "run", "--no-cache", code_file.name],
                cwd=str(self.temp_dir),
                capture_output=True,
                text=True,
                timeout=120
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

    async def analyze_with_gemini(self, questions_content: str, file_paths: List[str], max_retries: int, error_context: str = None) -> str:
        """Analyze all files with Gemini and execute code locally with retry logic"""
        for attempt in range(max_retries):
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

                logger.info(f"Analyzing {len(file_names)} files with Gemini (attempt {attempt + 1}/{max_retries})")

                system_prompt = analysis_prompt().strip()

                prompt = f"""
                        QUESTIONS TO ANSWER:
                        {questions_content}

                        EXACT FILE NAMES:
                        {file_names}

                        {error_context}
                        """

                contents = []
                if prompt and prompt.strip():
                    contents.append(prompt)
                if uploaded_files:
                    contents.extend(uploaded_files)

                response = await client.aio.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        system_instruction=system_prompt,
                        thinking_config=types.ThinkingConfig(thinking_budget=-1)
                    )
                )

                if response and hasattr(response, 'text') and response.text is not None:
                    generated_code = response.text.strip()
                    logger.info(f"Generated code: {len(generated_code)} characters")

                    execution_result = await self.execute_python_code(generated_code, file_paths)

                    for uploaded_file in uploaded_files:
                        try:
                            client.files.delete(name=uploaded_file.name)
                        except:
                            pass

                    if not execution_result.get("success", False) and attempt < max_retries - 1:
                        error_msg = execution_result.get("error", "Unknown execution error")
                        error_context = f"{generated_code}\n\n{str(error_msg)}"
                        logger.warning(f"Code execution failed: {error_msg}. Retrying with error context...")
                        return await self.analyze_with_gemini(questions_content, file_paths, max_retries - attempt - 1, error_context)

                    return execution_result
                else:
                    raise ValueError("Empty or null response from Gemini API")

            except Exception as e:
                for uploaded_file in uploaded_files:
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except:
                        pass

                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    wait_time = 10
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error in Gemini analysis after {max_retries} attempts: {str(e)}")
                    raise

        raise Exception(f"Failed to get response from Gemini after {max_retries} attempts")

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

            need_scraping = await self.check_scraping_requirement(questions_content)

            if urls and need_scraping:
                logger.info(f"Found URLs, starting parallel scraping process")
                scraped_results = await self.scrape_urls_parallel(urls, questions_content)
                saved_files.extend(scraped_results)

            result = await self.analyze_with_gemini(questions_content, saved_files, max_retries=2)

            if result.get("success", False):
                return result["result"]
            else:
                return {"error": result.get("error", "Unknown error occurred")}

        except Exception as e:
            logger.error(f"Error in process_request: {str(e)}")
            return {"error": str(e)}
