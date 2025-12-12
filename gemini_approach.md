# Gemini Agent Approach: Ab-initio to SQL Conversion

## Problem Statement

The primary goal is to convert a large Ab-initio MP file (approximately 2 million characters) into a single, validated SQL query. This task is complicated by several factors:
1.  **Large Input Size:** The Ab-initio file significantly exceeds the LLM's context window limit (100,000 tokens, roughly 400,000 characters).
2.  **Unstructured Ab-initio Code:** The Ab-initio MP file is unstructured, containing `xxparameter`, `xxgraph` sections, along with various joins and transformations, requiring intelligent parsing.
3.  **Additional Context:** Two supplementary files provide crucial information:
    *   Transformation details (source/target column names, join conditions).
    *   Column metadata (domain-specific information about columns).
4.  **LLM Constraints:** The agent must utilize a custom `TachyonBaseModel` (inheriting from Langchain BaseChatModel), which is capable of making tool calls but cannot produce structured output directly.
5.  **Logging Requirement:** The agent's message history must be incrementally logged to a file.
6.  **Validation:** The generated SQL query must be validated against a SQL database connection.

## Core Challenges Addressed

*   **Context Window Management:** How to process a 2 million character file with a 400,000 character LLM context limit.
*   **Ab-initio Parsing:** Accurately extracting and interpreting the complex, unstructured Ab-initio logic.
*   **SQL Generation:** Translating Ab-initio data flow, transformations, and joins into a correct, efficient, and single SQL query.
*   **Robust Validation:** Implementing a reliable mechanism to verify the generated SQL against a real database and provide actionable feedback for refinement.
*   **`TachyonBaseModel` Integration:** Designing tools and prompts that work effectively with a model that only supports tool calls and text output.
*   **Incremental Logging:** Ensuring all agent interactions and outputs are logged progressively.

## Proposed Solution: Iterative Agent with Specialized Tools

The solution employs an iterative agent architecture, leveraging specialized tools to handle file parsing, SQL generation, and database validation. The agent's workflow is designed to overcome the context window limitation and ensure the accuracy of the generated SQL through a feedback loop.

### 1. Pre-processing and Information Extraction

To address the large Ab-initio file size and its unstructured nature, a dedicated tool will be used to extract and summarize the critical components.

*   **Tool:** `AbInitioSectionExtractorTool`
    *   **Purpose:** To parse the entire Ab-initio MP file and extract relevant sections in a structured, concise string format suitable for the LLM.
    *   **Input:** The complete content of the Ab-initio MP file (as a string).
    *   **Functionality:**
        *   Utilizes regular expressions to identify and parse `xxparameter` blocks, `xxgraph` blocks, and within `xxgraph`, various components such as `INPUT FILE`, `OUTPUT FILE`, `REFORMAT`, `JOIN`, `FILTER`, `SORT`, etc.
        *   Extracts key attributes for each component, including component names, types, input/output ports, transformation logic (e.g., DML for `REFORMAT`), join conditions, file paths, and table names.
        *   Organizes this extracted information into a human-readable, structured string. This string is designed to be comprehensive yet compact, serving as the primary Ab-initio context for the LLM.
    *   **Output Format (Example):**
        ```
        --- Ab-initio Parameters ---
        PARAM_NAME_1 = 'value1'
        PARAM_NAME_2 = 'value2'

        --- Ab-initio Graph Components ---
        Component: INPUT FILE (source_data_1)
          Type: Input File
          File Path: ${PARAM_FILE_PATH_1}/file1.dat
          Output Port: out

        Component: REFORMAT (transform_logic_1)
          Type: Reformat
          Input Port: in (from source_data_1.out)
          Output Port: out
          DML:
            out.col1 :: in.src_col1;
            out.col2 :: in.src_col2 + 1;

        Component: JOIN (join_customers_orders)
          Type: Join
          Input Port: in0 (from transform_logic_1.out)
          Input Port: in1 (from source_data_2.out)
          Output Port: out
          Join Key: in0.customer_id == in1.customer_id
          Join Type: inner

        Component: OUTPUT FILE (target_data)
          Type: Output File
          File Path: ${PARAM_TARGET_PATH}/output.dat
          Input Port: in (from join_customers_orders.out)
        ```

### 2. Context Integration

The additional context files (`transformation_details.txt` and `column_metadata.txt`) will be read and parsed into Python dictionaries. These dictionaries will be passed directly to the `SQLGeneratorTool` to enrich the LLM's understanding during SQL construction.

### 3. SQL Generation

The core task of converting Ab-initio logic to SQL is handled by a specialized tool, guided by the LLM.

*   **Tool:** `SQLGeneratorTool`
    *   **Purpose:** To translate the extracted Ab-initio logic, transformation details, and column metadata into a SQL query.
    *   **Input:**
        *   The structured Ab-initio sections string (from `AbInitioSectionExtractorTool`).
        *   Parsed transformation details (dictionary).
        *   Parsed column metadata (dictionary).
        *   (Optional) Previous failed SQL query and error message (for iterative refinement).
    *   **Functionality (LLM-driven):** The LLM, through careful prompt engineering, will be instructed to:
        *   Identify source tables from `INPUT FILE` components and construct `FROM` clauses.
        *   Build `JOIN` clauses based on Ab-initio `JOIN` components and the provided transformation details.
        *   Formulate `SELECT` statements, including column transformations and expressions, derived from `REFORMAT` components and column metadata.
        *   Incorporate `WHERE` clauses for filtering logic.
        *   Handle Ab-initio parameters by substituting their values or integrating them into SQL conditions.
        *   Prioritize generating a single, comprehensive SQL query, utilizing Common Table Expressions (CTEs) for complex, multi-stage transformations to maintain readability and modularity.
    *   **Output:** The generated SQL query as a string.

### 4. SQL Validation and Iterative Refinement

A critical component of the solution is the ability to validate the generated SQL and iteratively refine it based on database feedback.

*   **Tool:** `SQLValidatorTool`
    *   **Purpose:** To execute the generated SQL query against a target database and provide detailed feedback.
    *   **Input:**
        *   `sql_query`: The SQL query string to be executed.
        *   `db_connection_string`: A connection string for the target SQL database (e.g., SQLite, PostgreSQL, MySQL).
    *   **Functionality:**
        *   Establishes a connection to the specified database.
        *   Executes the `sql_query`.
        *   Captures execution results (e.g., row count, schema) or any database-level errors.
    *   **Output (Structured Dictionary):**
        ```json
        {
            "success": bool,
            "error_message": str, # "(none)" if success is True
            "row_count": int,
            "schema": list[dict], # e.g., [{'name': 'col1', 'type': 'VARCHAR'}, {'name': 'col2', 'type': 'INT'}]
            "sample_data": list[dict] # e.g., [{'col1': 'val1', 'col2': 1}, {'col1': 'val2', 'col2': 2}]
        }
        ```
*   **Iterative Refinement Process:**
    1.  If `SQLValidatorTool` returns `success=False`, the agent receives the `error_message`.
    2.  The agent is then prompted again, this time with the original Ab-initio context, the *failed SQL query*, and the *specific error message*.
    3.  The LLM is instructed to analyze the error, identify the problematic part of the SQL, and generate a corrected SQL query using `SQLGeneratorTool`.
    4.  This generate-validate-refine loop continues for a predefined maximum number of retries or until a successful validation is achieved.

### 5. Incremental Logging

To meet the logging requirement, a dedicated `LoggerTool` is integrated throughout the agent's execution.

*   **Tool:** `LoggerTool`
    *   **Purpose:** To append messages to a specified log file incrementally.
    *   **Input:** `message` (string), `log_file_path` (string).
    *   **Functionality:** Opens the log file in append mode and writes the message, ensuring that the history is updated as the agent progresses.
*   **Integration Points:** `LoggerTool` calls are strategically placed at every significant step:
    *   After reading input files.
    *   After extracting Ab-initio sections.
    *   Before and after each `SQLGeneratorTool` call (logging the prompt and generated SQL).
    *   After each `SQLValidatorTool` call (logging validation results).
    *   When the final SQL query is successfully generated and validated.

### 6. Agent Orchestration

The agent will be orchestrated using Langchain's `AgentExecutor`.

*   **LLM:** The `TachyonBaseModel` will be used as the underlying language model.
*   **Prompt Engineering:** The agent's prompt will be carefully crafted to guide the LLM through the sequence of tool calls (extract -> generate -> validate -> refine) and to interpret the tool outputs effectively, especially given the `TachyonBaseModel`'s constraint of no structured output. The prompt will instruct the LLM to think step-by-step and use the tools appropriately.

## Addressing `TachyonBaseModel` Constraints

Since `TachyonBaseModel` cannot produce structured output directly, the design of the tools and the agent's interaction flow are crucial:
*   **String-based Communication:** All tools are designed to accept string inputs and return string outputs (or a string representation of structured data, as in `AbInitioSectionExtractorTool`).
*   **LLM as Orchestrator:** The LLM's primary role is to interpret the string outputs from tools, decide which tool to call next, and formulate the string arguments for those tools. Its "thought" process will be explicitly guided by the agent's prompt to ensure it follows the desired workflow.
*   **Tool-driven Structure:** The structured data (like validation results) is encapsulated within the tool's return value, which the LLM then interprets from its string representation.

## Validation Strategy

The validation strategy is central to ensuring the correctness of the generated SQL.

*   **Database Connection:** The `SQLValidatorTool` will connect to a real SQL database. For demonstration purposes, an in-memory SQLite database will be used, but the `db_connection_string` can be configured for any compatible database.
*   **Comprehensive Feedback:** The `SQLValidatorTool` returns not just a success/failure flag, but also an `error_message`, `row_count`, `schema`, and `sample_data`. This rich feedback allows the LLM to understand *why* a query failed and to verify the output structure and content of a successful query.
*   **Iterative Correction:** The agent's ability to re-attempt SQL generation based on validation errors is key to achieving a correct final query.

This comprehensive approach ensures that the challenges of large file size, complex Ab-initio logic, LLM constraints, and validation are effectively addressed, leading to a robust Ab-initio to SQL conversion agent.
