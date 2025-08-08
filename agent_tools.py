import os
import json
import asyncio
import subprocess
import sys
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import uuid
import pandas as pd
import numpy as np
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

@dataclass
class DataAnalystDeps:
    """Dependencies for the Data Analyst Agent"""
    temp_dir: str
    uploaded_files: List[str] = field(default_factory=list)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scraped_data: Dict[str, Any] = field(default_factory=dict)
    processed_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    generated_files: List[str] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)

def setup_web_scraping_tool(agent: Agent):
    """Setup web scraping tool using Crawl4AI"""
    
    @agent.tool
    async def scrape_website(
        ctx: RunContext[DataAnalystDeps], 
        url: str, 
        query: Optional[str] = None,
        extraction_type: str = "markdown"
    ) -> str:
        """
        Scrape a website using Crawl4AI with LLM-powered extraction.
        
        Args:
            ctx: Agent runtime context
            url: URL to scrape
            query: Optional query for targeted extraction
            extraction_type: Type of extraction (markdown, json, structured)
        
        Returns:
            Scraped and processed content
        """
        
        try:
            # Configure browser for Crawl4AI
            browser_config = BrowserConfig(
                browser_type="chromium",
                headless=True,
                verbose=True
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Configure extraction strategy
                if query:
                    # Use LLM extraction for targeted content
                    extraction_strategy = LLMExtractionStrategy(
                        provider="gemini-2.0-flash",
                        api_token=os.getenv("GEMINI_API_KEY"),
                        instruction=f"Extract information related to: {query}. Return structured data in JSON format."
                    )
                    
                    config = CrawlerRunConfig(
                        extraction_strategy=extraction_strategy,
                        cache_mode="bypass"
                    )
                else:
                    # Default markdown extraction
                    config = CrawlerRunConfig(
                        cache_mode="bypass"
                    )
                
                # Perform the scraping
                result = await crawler.arun(url=url, config=config)
                
                # Process and store results
                if extraction_type == "json" and result.extracted_content:
                    try:
                        structured_data = json.loads(result.extracted_content)
                        ctx.deps.scraped_data[url] = structured_data
                        content = json.dumps(structured_data, indent=2)
                    except json.JSONDecodeError:
                        content = result.extracted_content or result.markdown
                        ctx.deps.scraped_data[url] = {'content': content, 'type': 'text'}
                else:
                    content = result.markdown or result.cleaned_html
                    ctx.deps.scraped_data[url] = {
                        'content': content,
                        'type': extraction_type,
                        'query': query
                    }
                
                # Save to file for later use
                output_file = Path(ctx.deps.temp_dir) / f"scraped_{uuid.uuid4().hex[:8]}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                ctx.deps.generated_files.append(str(output_file))
                
                return f"Successfully scraped {url}. Content length: {len(content)} characters. Saved to: {output_file.name}"
                
        except Exception as e:
            return f"Error scraping {url} with Crawl4AI: {str(e)}"


def setup_code_generation_tool(agent: Agent):
    """Setup dynamic code generation and execution tools"""
    
    @agent.tool
    async def generate_and_execute_code(
        ctx: RunContext[DataAnalystDeps],
        task_description: str,
        code_language: str = "python",
        execute: bool = False
    ) -> str:
        """
        Generate code dynamically using LLM and optionally execute it
        
        Args:
            ctx: Agent runtime context
            task_description: Description of what the code should do
            code_language: Programming language (currently supports python)
            execute: Whether to execute the generated code
        
        Returns:
            Generated code and execution results
        """
        try:
            # Use the same LLM to generate code
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
            
            # Prepare context about available data
            data_context = ""
            if ctx.deps.processed_data:
                data_context = f"Available datasets: {list(ctx.deps.processed_data.keys())}"
                for name, df in ctx.deps.processed_data.items():
                    data_context += f"\n- {name}: shape {df.shape}, columns {list(df.columns)}"
            
            prompt = f"""
                        Generate {code_language} code for the following task:
                        {task_description}

                        Context:
                        - Working directory: {ctx.deps.temp_dir}
                        - {data_context}
                        - Generated files so far: {ctx.deps.generated_files}

                        Requirements:
                        1. Include all necessary imports
                        2. Add proper error handling
                        3. Use the available datasets if relevant
                        4. Include clear comments
                        5. Save any outputs to files in the working directory
                        6. Return a summary of what the code does

                        Generate clean, executable code only.
                    """
            
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            generated_code = response.text.strip()
            
            # Clean up code (remove markdown formatting if present)
            if generated_code.startswith('```python'):
                generated_code = generated_code[9:]
            if generated_code.endswith('```'):
                generated_code = generated_code[:-3]
            
            generated_code = generated_code.strip()
            
            # Save generated code
            code_file = Path(ctx.deps.temp_dir) / f"generated_code_{uuid.uuid4().hex[:8]}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            ctx.deps.generated_files.append(str(code_file))
            
            result = f"Generated code saved to: {code_file.name}\n\nCode:\n{generated_code}"
            
            if execute and code_language == "python":
                try:
                    # Execute the generated code in a controlled environment
                    exec_globals = {
                        'pd': pd,
                        'np': np,
                        'os': os,
                        'Path': Path,
                        'temp_dir': ctx.deps.temp_dir,
                        'processed_data': ctx.deps.processed_data,
                    }
                    
                    # Import common libraries for code execution
                    try:
                        import matplotlib.pyplot as plt
                        exec_globals['plt'] = plt
                    except ImportError:
                        pass
                    
                    try:
                        import seaborn as sns
                        exec_globals['sns'] = sns
                    except ImportError:
                        pass
                    
                    # Execute the code
                    exec(generated_code, exec_globals)
                    
                    result += "\n\nCode executed successfully!"
                    
                except Exception as e:
                    result += f"\n\nExecution error: {str(e)}"
            
            return result
            
        except Exception as e:
            return f"Error generating code: {str(e)}"
    
    @agent.tool  
    async def install_package(
        ctx: RunContext[DataAnalystDeps],
        package_name: str
    ) -> str:
        """
        Install Python packages dynamically
        
        Args:
            ctx: Agent runtime context
            package_name: Name of the package to install
        
        Returns:
            Installation result
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return f"Successfully installed {package_name}"
            else:
                return f"Failed to install {package_name}: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return f"Installation of {package_name} timed out"
        except Exception as e:
            return f"Error installing {package_name}: {str(e)}"