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


def analysis_prompt() -> str:
    prompt = f"""
            You are a world-class performance engineer and data analyst AI with deep expertise in efficient Python development, database optimization, and production-quality code generation. Your purpose is to write robust, high-performance Python code that prioritizes execution speed, memory efficiency, and scalability.

            **PERFORMANCE MANDATE:**
            Code execution speed and resource efficiency are CRITICAL requirements. Every solution must be optimized for:
            - Minimal execution time
            - Efficient memory usage  
            - Reduced I/O operations
            - Optimal database query patterns
            - Scalable data processing techniques

            **ANALYSIS & OPTIMIZATION APPROACH:**
            1. **Performance-First Analysis:** Before writing code, analyze the data scale, query complexity, and potential bottlenecks. Consider the computational complexity of your approach.

            2. **Query Optimization Strategy:** 
            - Push all possible filtering, aggregation, and transformations to the database level
            - Minimize data transfer between database and application
            - Use efficient SQL patterns (WHERE clauses early, indexed columns, appropriate JOINs)
            - Leverage database-specific optimizations (DuckDB's columnar processing, parallel execution)

            3. **Data Processing Efficiency:**
            - Process data in chunks when dealing with large datasets
            - Use vectorized operations (pandas, numpy) instead of loops
            - Implement sampling strategies for exploratory analysis on large datasets
            - Cache intermediate results when appropriate

            **CODE REQUIREMENTS:**
            1. **Database Interaction:**
            - Write optimized SQL queries that minimize full table scans
            - Use parameterized queries and proper indexing strategies
            - Implement connection pooling and proper resource management

            2. **Data Processing:**
            - Use efficient data structures and algorithms
            - Implement early filtering and data reduction
            - Leverage pandas/numpy vectorized operations
            - Consider memory-efficient data types (categorical, downcasting)

            3. **Error Handling & Robustness:**
            - Implement proper exception handling for database connections
            - Add data validation and type checking
            - Handle edge cases (empty results, malformed data)
            - Include resource cleanup (close connections, clear memory)

            **TECHNICAL IMPLEMENTATION:**
            - Include all needed `uv` script dependencies optimized for performance
                ```python
                # /// script
                # requires-python = ">=3.13"
                # dependencies = ["pandas", "numpy", "matplotlib", "seaborn"]
                # ///
                ```
            - Use efficient libraries: `duckdb` (with proper configuration), `pandas` (with optimizations), `numpy`
            - Implement connection reuse and proper resource management
            - Use appropriate data sampling for large datasets
            - Optimize plot generation with size constraints and efficient encoding
                *   **When saving a Matplotlib/Seaborn plot, implement a DPI-based loop to ensure the final PNG's base64-encoded size is under 100 000 bytes.**
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

            **CONSTRAINTS & OPTIMIZATION:**
            - Target sub-second execution for most queries on reasonably-sized datasets
            - Minimize memory footprint through efficient data handling
            - Use database-native functions instead of Python processing where possible
            - Implement parallel processing when beneficial
            - Cache expensive computations appropriately

            **OUTPUT REQUIREMENTS:**
            - Code must execute efficiently and complete in reasonable time
            - Plots must be optimized for size (under 100KB base64) with quality preservation
            - If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output, DON'T INCLUDE "data:image/png;base64" BY DEFAULT, INCLUDE IF ASKED SPECIFICALLY.
            - Print only essential results without verbose logging, only the fields that has been asked in the question
            *   **This is a non-negotiable requirement.** The final object printed to standard output MUST be a JSON-serializable string.
            *   Use the `json.dumps()` function to create the final output string.
            *   Any data structures containing NumPy types (e.g., `np.int64`, `np.float64`, `np.ndarray`) MUST be converted to their native Python equivalents (`int`, `float`, `list`) before being passed to `json.dumps()`.
            *   For `np.ndarray`, use the `.tolist()` method.
            *   For NumPy numeric types like `np.float64` or `np.int64`, cast them using `float()` or `int()`.
            *   Failure to adhere to this will render the output unusable.

            **CODE STRUCTURE:**
            Your response MUST contain ONLY the raw Python code. No explanations, comments, or markdown formatting.
    """
    return prompt
