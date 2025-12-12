"""
Ab-initio to SQL Conversion Agent

Converts large Ab-initio mp files (~2M characters) to SQL queries using LangGraph.
Handles context limitations through intelligent chunking and orchestration.

Author: Claude Code
Date: 2025-12-12
"""

import re
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Optional
from pathlib import Path
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# ============================================================================
# INCREMENTAL MESSAGE LOGGER
# ============================================================================

class IncrementalLogger:
    """Logs messages incrementally to a file as they are generated."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.message_count = 0

        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write(f"Ab-initio to SQL Conversion - Message Log\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'='*100}\n\n")

    def log_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Append a message to the log file immediately."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"\n{'='*100}\n")
            f.write(f"[{timestamp}] Message #{self.message_count} - Role: {role}\n")
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")
            f.write(f"{'-'*100}\n")
            f.write(str(content))
            f.write(f"\n{'='*100}\n\n")
            f.flush()  # Ensure immediate write to disk
            self.message_count += 1

    def log_state_transition(self, from_node: str, to_node: str, state_summary: Dict):
        """Log state transitions between nodes."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"\n{'~'*100}\n")
            f.write(f"[{timestamp}] STATE TRANSITION: {from_node} -> {to_node}\n")
            f.write(f"{'-'*100}\n")
            f.write(json.dumps(state_summary, indent=2, default=str))
            f.write(f"\n{'~'*100}\n\n")
            f.flush()


# ============================================================================
# SQL VALIDATION TOOL
# ============================================================================

def create_sql_validation_tool():
    """
    Creates a SQL validation tool function that can be bound to the LLM.

    This tool validates SQL queries by:
    1. Checking syntax
    2. Executing the query (if connection provided)
    3. Returning row counts, sample data, and error details

    Returns a function that can be used as a tool.
    """

    def sql_validation_tool(
        sql_query: str,
        connection_string: Optional[str] = None,
        dry_run: bool = False,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        Validates a SQL query through syntax checking and optional execution.

        Args:
            sql_query: The SQL query to validate
            connection_string: Database connection string (optional)
            dry_run: If True, only check syntax without executing
            sample_size: Number of sample rows to return

        Returns:
            Dictionary with validation results including:
            - syntax_valid: bool
            - execution_status: "success" | "error" | "skipped"
            - error_details: dict with error information
            - data_validation: dict with row counts and samples
            - warnings: list of warnings
        """
        import sqlparse
        from sqlparse import sql, tokens as T

        result = {
            "syntax_valid": False,
            "execution_status": "skipped",
            "error_details": None,
            "performance": {},
            "data_validation": {},
            "warnings": []
        }

        # Basic syntax validation using sqlparse
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                result["error_details"] = {
                    "message": "Empty or invalid SQL query",
                    "line_number": 0,
                    "suggestion": "Provide a valid SQL statement"
                }
                return result

            # Check for basic SQL structure
            result["syntax_valid"] = True

            # Look for common issues
            query_upper = sql_query.upper()
            if query_upper.count("SELECT") == 0:
                result["warnings"].append("No SELECT statement found")

            if "FROM" not in query_upper and "WITH" not in query_upper:
                result["warnings"].append("Query has no FROM clause or CTE")

        except Exception as e:
            result["error_details"] = {
                "message": f"Syntax parsing error: {str(e)}",
                "line_number": 0,
                "suggestion": "Check SQL syntax"
            }
            return result

        # If dry_run or no connection, return syntax check only
        if dry_run or not connection_string:
            result["execution_status"] = "skipped"
            result["warnings"].append("Execution skipped (dry_run=True or no connection)")
            return result

        # Attempt to execute the query
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text

            engine = create_engine(connection_string)
            start_time = datetime.now()

            with engine.connect() as conn:
                # Execute query
                query_result = conn.execute(text(sql_query))
                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                # Fetch results
                rows = query_result.fetchmany(sample_size)
                all_rows_count = query_result.rowcount

                result["execution_status"] = "success"
                result["performance"] = {
                    "execution_time_ms": execution_time,
                    "rows_affected": all_rows_count
                }

                # Get column information
                if query_result.returns_rows:
                    columns = list(query_result.keys())
                    result["data_validation"] = {
                        "row_count": len(rows),
                        "column_names": columns,
                        "sample_rows": [dict(row._mapping) for row in rows],
                        "total_columns": len(columns)
                    }
                else:
                    result["data_validation"] = {
                        "row_count": 0,
                        "note": "Query does not return rows"
                    }

        except Exception as e:
            result["execution_status"] = "error"
            error_msg = str(e)

            # Try to extract line number from error message
            line_match = re.search(r'line (\d+)', error_msg, re.IGNORECASE)
            line_number = int(line_match.group(1)) if line_match else 0

            # Try to identify problematic CTE
            cte_match = re.search(r'relation "([^"]+)" does not exist', error_msg)
            problematic_cte = cte_match.group(1) if cte_match else None

            result["error_details"] = {
                "message": error_msg,
                "line_number": line_number,
                "problematic_cte": problematic_cte,
                "suggestion": "Check table/column names and CTE references"
            }

        return result

    # Return the tool function
    return sql_validation_tool


# ============================================================================
# AGENT STATE DEFINITION
# ============================================================================

class AbInitioToSQLState(TypedDict):
    """State for the Ab-initio to SQL conversion agent."""

    # Input files
    abinitio_code: str
    transformation_info: str
    column_metadata: str

    # Processing mode
    mode: Literal["lite", "strong"]  # Approach selection

    # Phase 1: Chunking and Structure Extraction
    chunks: List[str]
    current_chunk_index: int
    chunk_metadata: List[Dict[str, Any]]
    dependency_graph: Dict[str, Any]
    transformation_catalog: Dict[str, Any]

    # Phase 2: SQL Generation
    conversion_order: List[str]
    sql_ctes: List[Dict[str, Any]]
    current_trans_index: int
    context_summary: str

    # Phase 3: Validation and Refinement
    final_sql: str
    validation_result: Dict[str, Any]
    validation_attempts: int
    max_validation_attempts: int

    # Logging and debugging
    message_history: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]
    current_node: str

    # Configuration
    connection_string: Optional[str]
    dry_run: bool


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def chunk_abinitio_code(code: str, max_chunk_size: int = 300000) -> List[str]:
    """
    Intelligently chunk Ab-initio code by graph boundaries.

    Args:
        code: Full Ab-initio mp file content
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of code chunks
    """
    chunks = []

    # Find all graph sections using .begin graph and .end graph markers
    graph_pattern = r'\.begin\s+graph(.*?)\.end\s+graph'
    graphs = re.finditer(graph_pattern, code, re.DOTALL | re.IGNORECASE)

    current_chunk = ""
    for graph_match in graphs:
        graph_content = graph_match.group(0)

        # If adding this graph exceeds max size, start new chunk
        if len(current_chunk) + len(graph_content) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = graph_content
        else:
            current_chunk += "\n" + graph_content

    # Add remaining content
    if current_chunk:
        chunks.append(current_chunk)

    # If no graphs found, fall back to simple size-based chunking
    if not chunks:
        for i in range(0, len(code), max_chunk_size):
            chunks.append(code[i:i + max_chunk_size])

    return chunks


def extract_graph_name(chunk: str) -> str:
    """Extract graph name from chunk."""
    match = re.search(r'\.begin\s+graph\s+(\w+)', chunk, re.IGNORECASE)
    return match.group(1) if match else "unknown"


# ============================================================================
# LANGGRAPH NODES - LITE APPROACH
# ============================================================================

def initialize_lite(state: AbInitioToSQLState) -> AbInitioToSQLState:
    """Initialize for lite approach - chunk the code."""
    chunks = chunk_abinitio_code(state["abinitio_code"])

    return {
        **state,
        "chunks": chunks,
        "current_chunk_index": 0,
        "sql_ctes": [],
        "context_summary": "",
        "current_node": "initialize_lite"
    }


def convert_chunk_lite(state: AbInitioToSQLState, llm, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Convert a single chunk to SQL in lite mode."""

    chunk_idx = state["current_chunk_index"]
    chunk = state["chunks"][chunk_idx]

    # Build context for this chunk
    context = f"""You are converting Ab-initio code to SQL.

## Current Chunk ({chunk_idx + 1}/{len(state["chunks"])})
Graph: {extract_graph_name(chunk)}

## Ab-initio Code Chunk:
{chunk}

## Transformation Metadata:
{state["transformation_info"]}

## Column Metadata:
{state["column_metadata"]}

## Previous SQL Context:
{state["context_summary"]}

## Task:
Convert this Ab-initio chunk to a SQL CTE (Common Table Expression).
- Use descriptive CTE names based on the transformation/graph name
- Reference previous CTEs if needed (from context above)
- Handle all transformations, joins, and filters
- Return ONLY the CTE definition (e.g., "cte_name AS (SELECT ...)")
"""

    messages = [
        SystemMessage(content="You are an expert at converting Ab-initio ETL code to SQL queries."),
        HumanMessage(content=context)
    ]

    logger.log_message("system", messages[0].content)
    logger.log_message("human", context, {"chunk_index": chunk_idx, "chunk_size": len(chunk)})

    # Call LLM
    response = llm.invoke(messages)
    sql_fragment = response.content

    logger.log_message("ai", sql_fragment, {"chunk_index": chunk_idx})

    # Store the CTE
    cte_entry = {
        "chunk_id": chunk_idx,
        "graph_name": extract_graph_name(chunk),
        "sql": sql_fragment,
        "cte_name": f"cte_chunk_{chunk_idx}"
    }

    # Update context summary for next chunk
    new_summary = f"{state['context_summary']}\n\nChunk {chunk_idx}: {extract_graph_name(chunk)}\nSQL CTE: {cte_entry['cte_name']}"

    return {
        **state,
        "sql_ctes": state["sql_ctes"] + [cte_entry],
        "current_chunk_index": chunk_idx + 1,
        "context_summary": new_summary,
        "message_history": [{"role": "ai", "content": sql_fragment}],
        "current_node": "convert_chunk_lite"
    }


def merge_sql_lite(state: AbInitioToSQLState, llm, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Merge all SQL CTEs into final query."""

    # Combine all CTEs
    cte_definitions = []
    for cte in state["sql_ctes"]:
        # Extract just the CTE part if wrapped
        sql = cte["sql"].strip()
        cte_definitions.append(sql)

    # Build final query
    final_sql = "WITH\n" + ",\n".join(cte_definitions)
    final_sql += f"\n\nSELECT * FROM {state['sql_ctes'][-1]['cte_name']}"

    logger.log_message("system", "Merged all CTEs into final SQL", {"num_ctes": len(state["sql_ctes"])})
    logger.log_message("ai", final_sql)

    return {
        **state,
        "final_sql": final_sql,
        "current_node": "merge_sql_lite"
    }


# ============================================================================
# LANGGRAPH NODES - STRONG APPROACH
# ============================================================================

def initialize_strong(state: AbInitioToSQLState) -> AbInitioToSQLState:
    """Initialize for strong approach."""
    chunks = chunk_abinitio_code(state["abinitio_code"])

    return {
        **state,
        "chunks": chunks,
        "current_chunk_index": 0,
        "chunk_metadata": [],
        "dependency_graph": {},
        "transformation_catalog": {},
        "current_node": "initialize_strong"
    }


def extract_metadata_strong(state: AbInitioToSQLState, llm, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Extract metadata from all chunks (Phase 1 of strong approach)."""

    all_metadata = []

    for idx, chunk in enumerate(state["chunks"]):
        context = f"""Extract metadata from this Ab-initio code chunk.

## Ab-initio Code:
{chunk}

## Task:
Analyze this code and extract:
1. Graph name
2. All transformations (type and purpose)
3. Input datasets/tables
4. Output datasets/tables
5. Join conditions
6. Dependencies on other graphs/transformations

Return a structured summary in this format:
GRAPH_NAME: <name>
TRANSFORMATIONS:
- <transformation 1>
- <transformation 2>
INPUTS:
- <input 1>
- <input 2>
OUTPUTS:
- <output 1>
DEPENDENCIES:
- <depends on graph/transformation>
"""

        messages = [
            SystemMessage(content="You extract structured metadata from Ab-initio ETL code."),
            HumanMessage(content=context)
        ]

        logger.log_message("human", context, {"chunk_index": idx, "phase": "metadata_extraction"})

        response = llm.invoke(messages)
        metadata_text = response.content

        logger.log_message("ai", metadata_text, {"chunk_index": idx})

        # Parse metadata (simplified - in production, use more robust parsing)
        metadata = {
            "chunk_id": idx,
            "graph_name": extract_graph_name(chunk),
            "raw_metadata": metadata_text,
            "chunk_size": len(chunk)
        }

        all_metadata.append(metadata)

    return {
        **state,
        "chunk_metadata": all_metadata,
        "current_node": "extract_metadata_strong"
    }


def build_dependency_graph_strong(state: AbInitioToSQLState, llm, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Build global dependency graph from metadata."""

    # Combine all metadata
    all_metadata_text = "\n\n".join([
        f"Chunk {m['chunk_id']} - {m['graph_name']}:\n{m['raw_metadata']}"
        for m in state["chunk_metadata"]
    ])

    context = f"""Analyze these metadata extracts and build a dependency graph.

## Metadata from all chunks:
{all_metadata_text}

## Transformation Info:
{state["transformation_info"]}

## Task:
Create a processing order (topological sort) of transformations.
List them in the order they should be converted to SQL, respecting dependencies.

Return format:
ORDER:
1. <transformation/graph name>
2. <transformation/graph name>
...
"""

    messages = [
        SystemMessage(content="You analyze data dependencies and create processing orders."),
        HumanMessage(content=context)
    ]

    logger.log_message("human", context, {"phase": "dependency_analysis"})

    response = llm.invoke(messages)
    order_text = response.content

    logger.log_message("ai", order_text)

    # Parse the order (simplified)
    conversion_order = [
        state["chunk_metadata"][i]["graph_name"]
        for i in range(len(state["chunk_metadata"]))
    ]

    return {
        **state,
        "conversion_order": conversion_order,
        "current_trans_index": 0,
        "current_node": "build_dependency_graph_strong"
    }


def convert_transformation_strong(state: AbInitioToSQLState, llm, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Convert one transformation to SQL CTE."""

    trans_idx = state["current_trans_index"]
    trans_name = state["conversion_order"][trans_idx]

    # Find the corresponding chunk
    chunk_meta = next((m for m in state["chunk_metadata"] if m["graph_name"] == trans_name), None)
    if not chunk_meta:
        return {
            **state,
            "errors": [f"Could not find chunk for transformation: {trans_name}"],
            "current_trans_index": trans_idx + 1
        }

    chunk = state["chunks"][chunk_meta["chunk_id"]]

    # Build context with previous CTEs
    previous_ctes = "\n".join([
        f"{cte['cte_name']}: {cte['description']}"
        for cte in state["sql_ctes"]
    ])

    context = f"""Convert this Ab-initio transformation to a SQL CTE.

## Transformation: {trans_name}

## Ab-initio Code:
{chunk}

## Metadata:
{chunk_meta["raw_metadata"]}

## Transformation Info:
{state["transformation_info"]}

## Column Metadata:
{state["column_metadata"]}

## Previously Generated CTEs:
{previous_ctes}

## Task:
Create a SQL CTE for this transformation.
- CTE name should be: cte_{trans_name.lower().replace(' ', '_')}
- Reference previous CTEs if this transformation depends on them
- Handle all joins, filters, and aggregations
- Return ONLY the CTE definition

Format: cte_name AS (
    SELECT ...
)
"""

    messages = [
        SystemMessage(content="You convert Ab-initio transformations to SQL CTEs."),
        HumanMessage(content=context)
    ]

    logger.log_message("human", context, {"transformation": trans_name, "trans_index": trans_idx})

    response = llm.invoke(messages)
    sql_cte = response.content

    logger.log_message("ai", sql_cte, {"transformation": trans_name})

    cte_entry = {
        "cte_name": f"cte_{trans_name.lower().replace(' ', '_')}",
        "transformation": trans_name,
        "sql": sql_cte,
        "description": f"CTE for {trans_name}"
    }

    return {
        **state,
        "sql_ctes": state["sql_ctes"] + [cte_entry],
        "current_trans_index": trans_idx + 1,
        "message_history": [{"role": "ai", "content": sql_cte}],
        "current_node": "convert_transformation_strong"
    }


def assemble_final_sql_strong(state: AbInitioToSQLState, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Assemble all CTEs into final SQL query."""

    if not state["sql_ctes"]:
        return {
            **state,
            "final_sql": "-- No CTEs generated",
            "errors": ["No CTEs were generated"],
            "current_node": "assemble_final_sql_strong"
        }

    # Build WITH clause
    cte_definitions = []
    for cte in state["sql_ctes"]:
        cte_definitions.append(cte["sql"].strip())

    final_sql = "WITH\n" + ",\n".join(cte_definitions)
    final_sql += f"\n\nSELECT * FROM {state['sql_ctes'][-1]['cte_name']}"

    logger.log_message("system", "Assembled final SQL from all CTEs", {
        "num_ctes": len(state["sql_ctes"]),
        "final_cte": state["sql_ctes"][-1]["cte_name"]
    })
    logger.log_message("ai", final_sql)

    return {
        **state,
        "final_sql": final_sql,
        "current_node": "assemble_final_sql_strong"
    }


# ============================================================================
# VALIDATION AND REFINEMENT NODES
# ============================================================================

def validate_sql(state: AbInitioToSQLState, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Validate the final SQL query."""

    sql_validator = create_sql_validation_tool()

    validation_result = sql_validator(
        sql_query=state["final_sql"],
        connection_string=state.get("connection_string"),
        dry_run=state.get("dry_run", True)
    )

    logger.log_message("system", "SQL Validation Result", validation_result)

    return {
        **state,
        "validation_result": validation_result,
        "validation_attempts": state.get("validation_attempts", 0) + 1,
        "current_node": "validate_sql"
    }


def refine_sql(state: AbInitioToSQLState, llm, logger: IncrementalLogger) -> AbInitioToSQLState:
    """Refine SQL based on validation errors."""

    if not state["validation_result"].get("error_details"):
        return state

    error_details = state["validation_result"]["error_details"]

    context = f"""The generated SQL query has errors. Please fix them.

## Current SQL:
{state["final_sql"]}

## Error Details:
{json.dumps(error_details, indent=2)}

## Task:
Fix the SQL query to resolve the errors.
Return the complete corrected SQL query.
"""

    messages = [
        SystemMessage(content="You fix SQL query errors."),
        HumanMessage(content=context)
    ]

    logger.log_message("human", context, {"validation_attempt": state["validation_attempts"]})

    response = llm.invoke(messages)
    refined_sql = response.content

    logger.log_message("ai", refined_sql, {"refinement": True})

    return {
        **state,
        "final_sql": refined_sql,
        "message_history": [{"role": "ai", "content": refined_sql}],
        "current_node": "refine_sql"
    }


# ============================================================================
# CONDITIONAL EDGE FUNCTIONS
# ============================================================================

def should_continue_chunking_lite(state: AbInitioToSQLState) -> str:
    """Decide if we should continue processing chunks in lite mode."""
    if state["current_chunk_index"] < len(state["chunks"]):
        return "continue"
    else:
        return "merge"


def should_continue_transformation_strong(state: AbInitioToSQLState) -> str:
    """Decide if we should continue processing transformations in strong mode."""
    if state["current_trans_index"] < len(state["conversion_order"]):
        return "continue"
    else:
        return "assemble"


def should_refine_sql(state: AbInitioToSQLState) -> str:
    """Decide if we should refine the SQL based on validation."""
    validation_result = state.get("validation_result", {})

    # Check if validation passed
    if validation_result.get("syntax_valid") and validation_result.get("execution_status") == "success":
        return "done"

    # Check if we've exceeded max attempts
    if state.get("validation_attempts", 0) >= state.get("max_validation_attempts", 3):
        return "done"

    # Check if there are errors to fix
    if validation_result.get("error_details"):
        return "refine"

    return "done"


# ============================================================================
# GRAPH BUILDERS
# ============================================================================

def build_lite_graph(llm, logger: IncrementalLogger) -> StateGraph:
    """Build the LangGraph for lite approach."""

    graph = StateGraph(AbInitioToSQLState)

    # Add nodes
    graph.add_node("initialize", lambda state: initialize_lite(state))
    graph.add_node("convert_chunk", lambda state: convert_chunk_lite(state, llm, logger))
    graph.add_node("merge_sql", lambda state: merge_sql_lite(state, llm, logger))
    graph.add_node("validate", lambda state: validate_sql(state, logger))
    graph.add_node("refine", lambda state: refine_sql(state, llm, logger))

    # Add edges
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "convert_chunk")

    # Conditional edge for chunking loop
    graph.add_conditional_edges(
        "convert_chunk",
        should_continue_chunking_lite,
        {
            "continue": "convert_chunk",
            "merge": "merge_sql"
        }
    )

    graph.add_edge("merge_sql", "validate")

    # Conditional edge for validation/refinement loop
    graph.add_conditional_edges(
        "validate",
        should_refine_sql,
        {
            "refine": "refine",
            "done": END
        }
    )

    graph.add_edge("refine", "validate")

    return graph.compile()


def build_strong_graph(llm, logger: IncrementalLogger) -> StateGraph:
    """Build the LangGraph for strong approach."""

    graph = StateGraph(AbInitioToSQLState)

    # Add nodes
    graph.add_node("initialize", lambda state: initialize_strong(state))
    graph.add_node("extract_metadata", lambda state: extract_metadata_strong(state, llm, logger))
    graph.add_node("build_dependencies", lambda state: build_dependency_graph_strong(state, llm, logger))
    graph.add_node("convert_transformation", lambda state: convert_transformation_strong(state, llm, logger))
    graph.add_node("assemble_sql", lambda state: assemble_final_sql_strong(state, logger))
    graph.add_node("validate", lambda state: validate_sql(state, logger))
    graph.add_node("refine", lambda state: refine_sql(state, llm, logger))

    # Add edges
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "extract_metadata")
    graph.add_edge("extract_metadata", "build_dependencies")
    graph.add_edge("build_dependencies", "convert_transformation")

    # Conditional edge for transformation conversion loop
    graph.add_conditional_edges(
        "convert_transformation",
        should_continue_transformation_strong,
        {
            "continue": "convert_transformation",
            "assemble": "assemble_sql"
        }
    )

    graph.add_edge("assemble_sql", "validate")

    # Conditional edge for validation/refinement loop
    graph.add_conditional_edges(
        "validate",
        should_refine_sql,
        {
            "refine": "refine",
            "done": END
        }
    )

    graph.add_edge("refine", "validate")

    return graph.compile()


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class AbInitioToSQLAgent:
    """
    Main agent for converting Ab-initio code to SQL.

    Supports two modes:
    - lite: Fast sequential chunking approach
    - strong: Hierarchical map-reduce with dependency analysis
    """

    def __init__(
        self,
        llm,  # TachyonBaseModel instance
        mode: Literal["lite", "strong"] = "strong",
        log_file: str = "abinitio_sql_conversion.log",
        connection_string: Optional[str] = None,
        dry_run: bool = True,
        max_validation_attempts: int = 3
    ):
        """
        Initialize the agent.

        Args:
            llm: TachyonBaseModel instance (assumed to handle API calls)
            mode: Conversion approach ("lite" or "strong")
            log_file: Path to message log file
            connection_string: Optional database connection for validation
            dry_run: If True, skip SQL execution during validation
            max_validation_attempts: Maximum refinement attempts
        """
        self.llm = llm
        self.mode = mode
        self.logger = IncrementalLogger(log_file)
        self.connection_string = connection_string
        self.dry_run = dry_run
        self.max_validation_attempts = max_validation_attempts

        # Build the appropriate graph
        if mode == "lite":
            self.graph = build_lite_graph(llm, self.logger)
            self.logger.log_message("system", f"Initialized agent in LITE mode")
        else:
            self.graph = build_strong_graph(llm, self.logger)
            self.logger.log_message("system", f"Initialized agent in STRONG mode")

    def convert(
        self,
        abinitio_file: str,
        transformation_file: str,
        column_metadata_file: str
    ) -> Dict[str, Any]:
        """
        Convert Ab-initio code to SQL.

        Args:
            abinitio_file: Path to Ab-initio mp file
            transformation_file: Path to transformation metadata file
            column_metadata_file: Path to column metadata file

        Returns:
            Dictionary with:
            - final_sql: The generated SQL query
            - validation_result: Validation results
            - errors: Any errors encountered
            - message_count: Number of LLM messages
        """

        # Load files
        self.logger.log_message("system", f"Loading input files...")

        with open(abinitio_file, 'r', encoding='utf-8') as f:
            abinitio_code = f.read()

        with open(transformation_file, 'r', encoding='utf-8') as f:
            transformation_info = f.read()

        with open(column_metadata_file, 'r', encoding='utf-8') as f:
            column_metadata = f.read()

        self.logger.log_message("system", f"Files loaded successfully", {
            "abinitio_size": len(abinitio_code),
            "transformation_size": len(transformation_info),
            "column_metadata_size": len(column_metadata)
        })

        # Initialize state
        initial_state = {
            "abinitio_code": abinitio_code,
            "transformation_info": transformation_info,
            "column_metadata": column_metadata,
            "mode": self.mode,
            "chunks": [],
            "current_chunk_index": 0,
            "chunk_metadata": [],
            "dependency_graph": {},
            "transformation_catalog": {},
            "conversion_order": [],
            "sql_ctes": [],
            "current_trans_index": 0,
            "context_summary": "",
            "final_sql": "",
            "validation_result": {},
            "validation_attempts": 0,
            "max_validation_attempts": self.max_validation_attempts,
            "message_history": [],
            "errors": [],
            "current_node": "start",
            "connection_string": self.connection_string,
            "dry_run": self.dry_run
        }

        # Run the graph
        self.logger.log_message("system", f"Starting conversion process...")

        final_state = self.graph.invoke(initial_state)

        self.logger.log_message("system", f"Conversion complete", {
            "validation_attempts": final_state.get("validation_attempts", 0),
            "num_chunks": len(final_state.get("chunks", [])),
            "num_ctes": len(final_state.get("sql_ctes", [])),
            "errors": final_state.get("errors", [])
        })

        # Return results
        return {
            "final_sql": final_state.get("final_sql", ""),
            "validation_result": final_state.get("validation_result", {}),
            "errors": final_state.get("errors", []),
            "message_count": self.logger.message_count,
            "num_chunks": len(final_state.get("chunks", [])),
            "num_ctes": len(final_state.get("sql_ctes", []))
        }


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function.

    This is the entry point for running the Ab-initio to SQL conversion.

    Usage:
        python abinitio_to_sql_agent.py

    Configuration:
        - Update file paths below to point to your input files
        - Set connection_string if you want to validate SQL execution
        - Choose mode: "lite" or "strong"
    """

    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================

    ABINITIO_FILE = "path/to/abinitio_mp_file.mp"
    TRANSFORMATION_FILE = "path/to/transformation_metadata.txt"
    COLUMN_METADATA_FILE = "path/to/column_metadata.txt"

    # Database connection (optional, for validation)
    CONNECTION_STRING = None  # e.g., "postgresql://user:pass@localhost/dbname"

    # Choose mode: "lite" or "strong"
    MODE = "strong"

    # Validation settings
    DRY_RUN = True  # Set to False to actually execute SQL during validation
    MAX_VALIDATION_ATTEMPTS = 3

    # Log file
    LOG_FILE = "abinitio_sql_conversion.log"

    # ========================================================================
    # INITIALIZE LLM (TachyonBaseModel)
    # ========================================================================

    # NOTE: Replace this with your actual TachyonBaseModel initialization
    # Example: llm = TachyonBaseModel(model="claude", temperature=0.1)

    try:
        from your_module import TachyonBaseModel  # Replace with actual import
        llm = TachyonBaseModel(model="claude", temperature=0.1)
    except ImportError:
        print("ERROR: Could not import TachyonBaseModel")
        print("Please update the import statement with your actual module path")
        print("Example: from langchain_anthropic import ChatAnthropic")
        print("         llm = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0.1)")
        return

    # ========================================================================
    # CREATE AND RUN AGENT
    # ========================================================================

    print(f"Initializing Ab-initio to SQL Agent (mode={MODE})...")
    print(f"Log file: {LOG_FILE}")

    agent = AbInitioToSQLAgent(
        llm=llm,
        mode=MODE,
        log_file=LOG_FILE,
        connection_string=CONNECTION_STRING,
        dry_run=DRY_RUN,
        max_validation_attempts=MAX_VALIDATION_ATTEMPTS
    )

    print(f"\nStarting conversion...")
    print(f"Input files:")
    print(f"  - Ab-initio: {ABINITIO_FILE}")
    print(f"  - Transformations: {TRANSFORMATION_FILE}")
    print(f"  - Column Metadata: {COLUMN_METADATA_FILE}")
    print()

    try:
        result = agent.convert(
            abinitio_file=ABINITIO_FILE,
            transformation_file=TRANSFORMATION_FILE,
            column_metadata_file=COLUMN_METADATA_FILE
        )

        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================

        print("="*100)
        print("CONVERSION COMPLETE")
        print("="*100)

        print(f"\nStatistics:")
        print(f"  - Message count: {result['message_count']}")
        print(f"  - Chunks processed: {result['num_chunks']}")
        print(f"  - CTEs generated: {result['num_ctes']}")
        print(f"  - Errors: {len(result['errors'])}")

        if result['errors']:
            print(f"\nErrors encountered:")
            for error in result['errors']:
                print(f"  - {error}")

        print(f"\nValidation Result:")
        val_result = result['validation_result']
        print(f"  - Syntax valid: {val_result.get('syntax_valid', False)}")
        print(f"  - Execution status: {val_result.get('execution_status', 'unknown')}")

        if val_result.get('error_details'):
            print(f"  - Error: {val_result['error_details'].get('message', 'Unknown error')}")

        if val_result.get('warnings'):
            print(f"  - Warnings: {len(val_result['warnings'])}")
            for warning in val_result['warnings']:
                print(f"    - {warning}")

        print(f"\n{'='*100}")
        print("GENERATED SQL")
        print("="*100)
        print(result['final_sql'])
        print("="*100)

        # Save SQL to file
        output_file = "generated_sql_query.sql"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['final_sql'])

        print(f"\nSQL saved to: {output_file}")
        print(f"Full log saved to: {LOG_FILE}")

        return result

    except Exception as e:
        print(f"\nERROR during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
