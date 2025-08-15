from typing import List


def scraping_prompt(questions_content: str) -> str:
    prompt = f"""
            Understand the following question and just answer "True" or "False" if the text contains any link that needs to be scraped in order to solve the given questions.
            DO NOT SOLVE THE QUESTION

            QUESTIONS:
            {questions_content}
            """
    return prompt


def cleaning_prompt(url: str, raw_content: str, questions_content: str) -> str:
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
    return prompt


def analysis_prompt(questions_content: str, file_names: List[str], error_context: str = None) -> str:
    prompt = f"""
            You are a world-class data analyst AI who is also an expert Python developer with deep knowledge of data serialization standards for production systems. Your purpose is to write robust, production-quality Python code to solve a user's question based on the data files they provide. 
            You must follow these instructions meticulously:
            1.  **Analyze the Request:** Carefully read the user's question and examine the previews of all provided files (text, CSV, images, etc.) to understand the context and requirements fully.
            2.  **Think Step-by-Step:** Before writing code, formulate a clear plan inside a `<thought>` block. Consider data loading, necessary cleaning (handling missing values, correcting data types), analysis steps, and the final output format. Your thought process is for your own guidance and should not be in the final Python code.
            3.  If the question contains one or several links and 
                *   If scraping is mentioned for the link, assume the scraped data is already uploaded to you. Do NOT scrape anything.
                *   If data needed to fetched/ downloaded, write code to source it in the python script.
            
            4.  **Write High-Quality Python Code:**
                *   The code must be pure Python and executable. DO NOT RUN OR EXECUTE THE CODE YOURSELF. DO NOT WRITE ANY COMMENTS IN CODE.
                *   Include all needed `uv` script dependencies at the top of the code, for example:
                    ```
                    # /// script
                    # requires-python = ">=3.13"
                    # dependencies = ["pandas", "numpy", "matplotlib", "seaborn"]
                    # ///
                    ```
                *   Refer to files by their exact filenames. Do not assume file paths; instead, robustly implement finding the filepath by filename (e.g., using `os.walk`) to avoid `FileNotFoundError`.
                *   Perform data cleaning and preprocessing. Do not make assumptions about data quality. Check for and handle inconsistencies.
                *   Your code must print the final answer(s) to standard output.
                *   Your code must print only the actual values of final answer(s), without any filler words or sentence.
                *   If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output.
                *   **When saving a Matplotlib/Seaborn plot, implement a DPI-based loop to ensure the final PNG’s base64-encoded size is under 100 000 bytes.**
                    Minimal example:
                    ```
                    import io, base64, matplotlib.pyplot as plt
                    def save_plot_under_limit(fig, fname, limit=100_000):
                        dpi = 120
                        while dpi >= 20:
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
                            b = base64.b64encode(buf.getvalue())
                            if len(b) <= limit:
                                open(fname, 'wb').write(buf.getvalue())
                                print("data:image/png;base64," + b.decode())
                                return
                            dpi -= 10
                        raise ValueError("Cannot meet size limit")
                    ```
            5.  **Output Serialization Mandate:**
                *   **This is a non-negotiable requirement.** The final object printed to standard output MUST be a JSON-serializable string.
                *   Use the `json.dumps()` function to create the final output string.
                *   Any data structures containing NumPy types (e.g., `np.int64`, `np.float64`, `np.ndarray`) MUST be converted to their native Python equivalents (`int`, `float`, `list`) before being passed to `json.dumps()`.
                *   For `np.ndarray`, use the `.tolist()` method.
                *   For NumPy numeric types like `np.float64` or `np.int64`, cast them using `float()` or `int()`.
                *   Failure to adhere to this will render the output unusable.
            6.  **Final Output:** Your response MUST contain ONLY the raw Python code. Do not include any explanations, comments, or markdown formatting like ``````. Just the code itself.
            
            QUESTIONS TO ANSWER:
            {questions_content}

            EXACT FILE NAMES:
            {file_names}

            {error_context}
            """
    return prompt
