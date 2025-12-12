import re
import json
import sqlite3
from typing import List, Dict, Any, Optional, Type

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

# --- Mock TachyonBaseModel (as per constraints) ---
class TachyonBaseModel(BaseChatModel):
    """
    A mock implementation of a custom Langchain BaseChatModel.
    As per constraints, it inherits from BaseChatModel and is capable of making tool calls
    but cannot produce structured output directly.
    """
    model: str = "claude" # Placeholder for the actual model name
    temperature: float = 0.8 # Placeholder for temperature setting

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> Any:
        # This is a mock implementation. In a real scenario, this would
        # interact with the actual Tachyon LLM API.
        # For the purpose of this agent, we will simulate the LLM's behavior
        # by having it call tools based on the prompt.
        # The agent executor will handle the actual tool calling logic.
        raise NotImplementedError("This is a mock model and should not be called directly for generation.")

    @property
    def _llm_type(self) -> str:
        return "tachyon_mock_chat"

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> BaseMessage:
        # This method is typically used for direct chat completion.
        # For an agent, the _generate method (or the agent executor's logic)
        # is more relevant for tool calling.
        # We'll keep this simple as the agent executor will drive the tool calls.
        return AIMessage(content="Mock response from TachyonBaseModel.")

# --- Custom Tools ---

class LoggerTool(BaseTool):
    name: str = "logger_tool"
    description: str = "Appends a message to a specified log file incrementally."
    log_file_path: str

    def _run(self, message: str) -> str:
        try:
            with open(self.log_file_path, "a") as f:
                f.write(message + "\n")
            return f"Message logged to {self.log_file_path}"
        except Exception as e:
            return f"Error logging message: {e}"

    async def _arun(self, message: str) -> str:
        return self._run(message) # Simple async for now

class AbInitioSectionExtractorTool(BaseTool):
    name: str = "ab_initio_section_extractor_tool"
    description: str = (
        "Parses a large Ab-initio MP file content and extracts key sections "
        "(parameters, graph components, transformations, joins) into a structured string. "
        "Input is the full Ab-initio file content as a string."
    )

    def _run(self, ab_initio_content: str) -> str:
        extracted_info = []

        # Extract xxparameter sections
        param_pattern = re.compile(r"xxparameter\s+(\w+)\s*:\s*(.*?)(?=\nxxparameter|\nxxgraph|\Z)", re.DOTALL)
        params = param_pattern.findall(ab_initio_content)
        if params:
            extracted_info.append("--- Ab-initio Parameters ---")
            for name, value in params:
                extracted_info.append(f"{name} = {value.strip()}")
            extracted_info.append("")

        # Extract xxgraph sections and components within them
        graph_pattern = re.compile(r"xxgraph\s+(\w+)\s*:\s*(.*?)(?=\nxxgraph|\Z)", re.DOTALL)
        graphs = graph_pattern.findall(ab_initio_content)

        if graphs:
            extracted_info.append("--- Ab-initio Graph Components ---")
            for graph_name, graph_content in graphs:
                extracted_info.append(f"Graph: {graph_name}")

                # Component pattern (e.g., INPUT FILE, REFORMAT, JOIN, OUTPUT FILE)
                component_pattern = re.compile(r"(\w+\s+FILE|\w+)\s+(\w+)\s*:\s*(.*?)(?=\n\s*\w+\s+FILE|\n\s*\w+\s+\w+\s*:|\Z)", re.DOTALL)
                components = component_pattern.findall(graph_content)

                for comp_type_raw, comp_name, comp_details in components:
                    comp_type = comp_type_raw.strip()
                    extracted_info.append(f"  Component: {comp_type} ({comp_name})")
                    extracted_info.append(f"    Type: {comp_type}")

                    # Extract details based on component type
                    if "INPUT FILE" in comp_type or "OUTPUT FILE" in comp_type:
                        file_path_match = re.search(r"file\s+=\s*\"(.*?)\"", comp_details)
                        if file_path_match:
                            extracted_info.append(f"    File Path: {file_path_match.group(1)}")
                        port_match = re.search(r"(in|out)\s+port", comp_details)
                        if port_match:
                            extracted_info.append(f"    Port: {port_match.group(1)}")
                    elif "REFORMAT" in comp_type:
                        dml_match = re.search(r"dml\s*=\s*\"\"\"(.*?)\"\"\"", comp_details, re.DOTALL)
                        if dml_match:
                            extracted_info.append("    DML:")
                            for line in dml_match.group(1).split('\n'):
                                extracted_info.append(f"      {line.strip()}")
                        in_port_match = re.search(r"in\s+port\s+=\s*(\w+)", comp_details)
                        if in_port_match:
                            extracted_info.append(f"    Input Port: {in_port_match.group(1)}")
                        out_port_match = re.search(r"out\s+port\s+=\s*(\w+)", comp_details)
                        if out_port_match:
                            extracted_info.append(f"    Output Port: {out_port_match.group(1)}")
                    elif "JOIN" in comp_type:
                        join_key_match = re.search(r"key\s*=\s*\"(.*?)\"", comp_details)
                        if join_key_match:
                            extracted_info.append(f"    Join Key: {join_key_match.group(1)}")
                        join_type_match = re.search(r"type\s*=\s*\"(.*?)\"", comp_details)
                        if join_type_match:
                            extracted_info.append(f"    Join Type: {join_type_match.group(1)}")
                        for i in range(5): # Check for multiple input ports
                            in_port_match = re.search(rf"in{i}\s+port\s+=\s*(\w+)", comp_details)
                            if in_port_match:
                                extracted_info.append(f"    Input Port {i}: {in_port_match.group(1)}")
                        out_port_match = re.search(r"out\s+port\s+=\s*(\w+)", comp_details)
                        if out_port_match:
                            extracted_info.append(f"    Output Port: {out_port_match.group(1)}")
                    # Add more component types as needed (e.g., FILTER, SORT)

                    extracted_info.append("") # Newline for readability

        return "\n".join(extracted_info)

    async def _arun(self, ab_initio_content: str) -> str:
        return self._run(ab_initio_content)

class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator_tool"
    description: str = (
        "Generates a SQL query based on extracted Ab-initio logic, transformation details, "
        "column metadata, and optionally, a previous failed SQL query with an error message. "
        "Input should be a JSON string containing 'ab_initio_summary', 'transformation_details', "
        "'column_metadata', 'previous_sql' (optional), and 'error_message' (optional)."
    )

    def _run(self, input_json_string: str) -> str:
        input_data = json.loads(input_json_string)
        ab_initio_summary = input_data["ab_initio_summary"]
        transformation_details = input_data["transformation_details"]
        column_metadata = input_data["column_metadata"]
        previous_sql = input_data.get("previous_sql", "")
        error_message = input_data.get("error_message", "")

        # This is where the LLM would do the heavy lifting of generating SQL.
        # For this mock, we'll return a placeholder or a simple SQL based on input.
        # In a real scenario, the LLM would receive a prompt constructed from these inputs.

        prompt_parts = [
            "You are an expert in converting Ab-initio graphs to SQL queries.",
            "Given the following Ab-initio graph summary, transformation details, and column metadata, generate a single, comprehensive SQL query.",
            "Use Common Table Expressions (CTEs) for multi-stage transformations to ensure readability and correctness.",
            "--- Ab-initio Graph Summary ---",
            ab_initio_summary,
            "--- Transformation Details ---",
            json.dumps(transformation_details, indent=2),
            "--- Column Metadata ---",
            json.dumps(column_metadata, indent=2),
        ]

        if previous_sql and error_message:
            prompt_parts.append("--- Previous Failed SQL Query ---")
            prompt_parts.append(previous_sql)
            prompt_parts.append("--- Error Message ---")
            prompt_parts.append(error_message)
            prompt_parts.append("Analyze the error and generate a corrected SQL query.")
        else:
            prompt_parts.append("Generate the initial SQL query.")

        # Simulate LLM generating SQL. In a real agent, the LLM would be called here.
        # For now, we return a very basic placeholder.
        # The actual LLM call would be part of the agent's _generate method,
        # which then uses this tool's output.
        # This tool's _run method is essentially a wrapper for the LLM's SQL generation capability.
        # The agent's prompt will guide the LLM to output the SQL directly as the tool's result.

        # Example of a very basic SQL generation logic (LLM would do this more intelligently)
        if "INPUT FILE (source_data_1)" in ab_initio_summary:
            return "SELECT * FROM source_data_1;"
        return "SELECT 'Generated SQL Placeholder' AS result;"

    async def _arun(self, input_json_string: str) -> str:
        return self._run(input_json_string)

class SQLValidatorTool(BaseTool):
    name: str = "sql_validator_tool"
    description: str = (
        "Executes a SQL query against a database and returns validation results. "
        "Input is a JSON string containing 'sql_query' and 'db_connection_string'."
        "Returns a JSON string with 'success', 'error_message', 'row_count', 'schema', and 'sample_data'."
    )

    def _run(self, input_json_string: str) -> str:
        input_data = json.loads(input_json_string)
        sql_query = input_data["sql_query"]
        db_connection_string = input_data["db_connection_string"]

        conn = None
        try:
            conn = sqlite3.connect(db_connection_string)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            row_count = len(rows)

            schema = []
            if cursor.description:
                for col in cursor.description:
                    schema.append({"name": col[0], "type": str(col[1])})

            sample_data = []
            if rows:
                # Get up to 5 sample rows
                for i, row in enumerate(rows[:5]):
                    sample_data.append(dict(zip([col[0] for col in cursor.description], row)))

            return json.dumps({
                "success": True,
                "error_message": "(none)",
                "row_count": row_count,
                "schema": schema,
                "sample_data": sample_data
            })
        except sqlite3.Error as e:
            return json.dumps({
                "success": False,
                "error_message": str(e),
                "row_count": 0,
                "schema": [],
                "sample_data": []
            })
        finally:
            if conn:
                conn.close()

    async def _arun(self, input_json_string: str) -> str:
        return self._run(input_json_string)

# --- Agent Definition ---

def create_ab_initio_to_sql_agent(
    llm: TachyonBaseModel,
    tools: List[BaseTool],
    log_file: str,
    verbose: bool = True
) -> AgentExecutor:
    """
    Creates an AgentExecutor for Ab-initio to SQL conversion.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Ab-initio to SQL converter agent. Your goal is to convert complex Ab-initio graphs into a single, correct, and validated SQL query. You have access to specialized tools for extracting Ab-initio sections, generating SQL, and validating SQL against a database. Log all your steps and thoughts using the logger_tool."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # The create_react_agent function helps in setting up the agent with ReAct prompting.
    # The LLM will be prompted to output thoughts and then tool actions.
    agent: Runnable = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True # Important for robust agent behavior
    )
    return agent_executor

# --- Main Execution Logic ---

if __name__ == "__main__":
    # Define file paths (placeholders - replace with actual paths)
    AB_INITIO_MP_FILE = "/home/nitish/Documents/github/Langchain/ab_initio_example.mp" # Create this file for testing
    TRANSFORMATION_DETAILS_FILE = "/home/nitish/Documents/github/Langchain/transformation_details.txt" # Create this file for testing
    COLUMN_METADATA_FILE = "/home/nitish/Documents/github/Langchain/column_metadata.txt" # Create this file for testing
    LOG_FILE = "/home/nitish/Documents/github/Langchain/agent_log.txt"
    DB_CONNECTION_STRING = "sqlite:///:memory:" # In-memory SQLite for demonstration

    # Create dummy input files for testing
    with open(AB_INITIO_MP_FILE, "w") as f:
        f.write("""
xxparameter MY_PARAM : "some_value"
xxparameter INPUT_FILE_PATH : "/data/raw"
xxparameter OUTPUT_FILE_PATH : "/data/processed"

xxgraph my_ab_initio_graph :
  INPUT FILE source_customers :
    file = "${INPUT_FILE_PATH}/customers.dat"
    layout = "customer_id string(10), name string(50), age integer(3)"
    out port

  REFORMAT transform_customer_data :
    in port = source_customers.out
    out port
    dml = """
out.customer_id :: in.customer_id;
out.full_name :: string_concat(in.name, " ", "Doe");
out.age_group :: if (in.age < 30) "Young" else "Old";
"""

  INPUT FILE source_orders :
    file = "${INPUT_FILE_PATH}/orders.dat"
    layout = "order_id string(10), customer_id string(10), order_date date, amount decimal(10,2)"
    out port

  JOIN join_customers_orders :
    in0 port = transform_customer_data.out
    in1 port = source_orders.out
    out port
    key = "customer_id"
    type = "inner"

  OUTPUT FILE target_output :
    file = "${OUTPUT_FILE_PATH}/customer_orders.dat"
    layout = "customer_id string(10), full_name string(100), age_group string(10), order_id string(10), order_date date, amount decimal(10,2)"
    in port = join_customers_orders.out
        """)

    with open(TRANSFORMATION_DETAILS_FILE, "w") as f:
        f.write("""
# Source to Target Column Mappings and Joins
customers.customer_id -> final_output.customer_id
customers.name -> final_output.full_name (concatenated with "Doe")
customers.age -> final_output.age_group (conditional logic)
orders.order_id -> final_output.order_id
orders.customer_id -> final_output.customer_id
orders.order_date -> final_output.order_date
orders.amount -> final_output.amount

JOIN: customers.customer_id = orders.customer_id
        """)

    with open(COLUMN_METADATA_FILE, "w") as f:
        f.write("""
# Column Metadata
customer_id: Unique identifier for customers. String.
name: Customer's first name. String.
age: Customer's age in years. Integer.
order_id: Unique identifier for orders. String.
order_date: Date of the order. Date.
amount: Total amount of the order. Decimal.
full_name: Derived column, full name of the customer. String.
age_group: Derived column, categorizes age. String.
        """)

    # Read input files
    with open(AB_INITIO_MP_FILE, "r") as f:
        ab_initio_content = f.read()
    with open(TRANSFORMATION_DETAILS_FILE, "r") as f:
        transformation_details_raw = f.read()
    with open(COLUMN_METADATA_FILE, "r") as f:
        column_metadata_raw = f.read()

    # Parse transformation details and column metadata into dictionaries
    # (This parsing can be more sophisticated for real-world scenarios)
    transformation_details = {}
    for line in transformation_details_raw.split('\n'):
        if "->" in line:
            src, tgt = line.split("->")
            transformation_details[src.strip()] = tgt.strip()
        elif "JOIN:" in line:
            transformation_details["JOIN_CONDITION"] = line.replace("JOIN:", "").strip()

    column_metadata = {}
    for line in column_metadata_raw.split('\n'):
        if ":" in line and not line.startswith("#"):
            col, meta = line.split(":")
            column_metadata[col.strip()] = meta.strip()

    # Initialize LLM and tools
    llm = TachyonBaseModel(model="claude", temperature=0.7) # Using the mock model
    logger_tool = LoggerTool(log_file_path=LOG_FILE)
    ab_initio_extractor_tool = AbInitioSectionExtractorTool()
    sql_generator_tool = SQLGeneratorTool()
    sql_validator_tool = SQLValidatorTool()

    tools = [logger_tool, ab_initio_extractor_tool, sql_generator_tool, sql_validator_tool]

    # Create the agent
    agent_executor = create_ab_initio_to_sql_agent(llm, tools, LOG_FILE)

    # Initial agent input
    agent_input = {
        "input": (
            "Convert the provided Ab-initio MP file into a single SQL query. "
            "Use the transformation details and column metadata for accurate mapping. "
            "Validate the generated SQL using the SQLValidatorTool. "
            "Here are the contents of the files:\n\n"
            f"--- Ab-initio MP File ---\n{ab_initio_content}\n\n"
            f"--- Transformation Details ---\n{transformation_details_raw}\n\n"
            f"--- Column Metadata ---\n{column_metadata_raw}\n"
        ),
        "chat_history": [],
    }

    print(f"Starting agent execution. Log file: {LOG_FILE}")
    logger_tool._run(f"Agent started at: {__import__('datetime').datetime.now()}")
    logger_tool._run(f"Initial input: {agent_input['input']}")

    final_sql_query = None
    max_retries = 3
    current_retry = 0

    while current_retry < max_retries:
        try:
            # The agent will use the tools to process the input
            # and eventually call sql_generator_tool and sql_validator_tool
            # The output of the agent will be the final SQL query or an error message.
            # Since TachyonBaseModel is a mock, we need to simulate the agent's steps.

            # Step 1: Extract Ab-initio sections
            logger_tool._run("Agent thought: Extracting Ab-initio sections...")
            ab_initio_summary = ab_initio_extractor_tool._run(ab_initio_content)
            logger_tool._run(f"Extracted Ab-initio Summary:\n{ab_initio_summary}")

            # Step 2: Generate SQL
            logger_tool._run("Agent thought: Generating SQL query...")
            sql_generation_input = json.dumps({
                "ab_initio_summary": ab_initio_summary,
                "transformation_details": transformation_details,
                "column_metadata": column_metadata,
                "previous_sql": "", # No previous SQL yet
                "error_message": ""
            })
            generated_sql = sql_generator_tool._run(sql_generation_input)
            logger_tool._run(f"Generated SQL:\n{generated_sql}")

            # Step 3: Validate SQL
            logger_tool._run("Agent thought: Validating SQL query...")
            sql_validation_input = json.dumps({
                "sql_query": generated_sql,
                "db_connection_string": DB_CONNECTION_STRING
            })
            validation_result_json = sql_validator_tool._run(sql_validation_input)
            validation_result = json.loads(validation_result_json)
            logger_tool._run(f"SQL Validation Result:\n{json.dumps(validation_result, indent=2)}")

            if validation_result["success"]:
                final_sql_query = generated_sql
                logger_tool._run("Agent thought: SQL query validated successfully!")
                print("\n--- Final SQL Query (Validated) ---")
                print(final_sql_query)
                print("\n--- Validation Details ---")
                print(f"Row Count: {validation_result['row_count']}")
                print(f"Schema: {validation_result['schema']}")
                print(f"Sample Data: {validation_result['sample_data']}")
                break
            else:
                logger_tool._run(f"Agent thought: SQL validation failed. Error: {validation_result['error_message']}")
                print(f"SQL validation failed. Retrying... Error: {validation_result['error_message']}")
                current_retry += 1
                # For a real agent, the LLM would be prompted again with the error
                # and the agent_executor would handle the retry logic.
                # Here, we're simulating a simple retry.
                # In a more advanced mock, we could have sql_generator_tool return a "corrected" SQL.
                if current_retry < max_retries:
                    logger_tool._run(f"Agent thought: Retrying SQL generation with error feedback. Retry {current_retry}/{max_retries}")
                    # Simulate LLM trying to correct the SQL
                    # For this mock, we'll just try to generate again,
                    # but a real LLM would use the error_message to refine.
                    sql_generation_input = json.dumps({
                        "ab_initio_summary": ab_initio_summary,
                        "transformation_details": transformation_details,
                        "column_metadata": column_metadata,
                        "previous_sql": generated_sql,
                        "error_message": validation_result['error_message']
                    })
                    generated_sql = sql_generator_tool._run(sql_generation_input)
                    logger_tool._run(f"Generated (Retry {current_retry}) SQL:\n{generated_sql}")
                    # Then loop back to validation
                else:
                    logger_tool._run("Agent thought: Max retries reached. Could not generate valid SQL.")
                    print("\n--- Failed to generate valid SQL after multiple retries ---")
                    print(f"Last generated SQL:\n{generated_sql}")
                    print(f"Last error:\n{validation_result['error_message']}")
                    break

        except Exception as e:
            logger_tool._run(f"Agent encountered an unexpected error: {e}")
            print(f"An unexpected error occurred: {e}")
            break

    if not final_sql_query:
        print("\nAgent failed to produce a validated SQL query.")
    logger_tool._run(f"Agent finished at: {__import__('datetime').datetime.now()}")
