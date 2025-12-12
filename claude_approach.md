# Ab-initio to SQL Conversion Agent - Approaches

## Problem Statement
Convert ~2 million characters of Ab-initio code (mp file) to SQL query, but LLM context limit is only ~400,000 characters (100k tokens).

**Input Files:**
1. Ab-initio mp file (~2M chars) - Contains graphs, transformations, joins, parameters
2. Transformation metadata - Source/target columns, join info (without join types)
3. Column domain metadata - Column descriptions and business context

**Constraints:**
- Must use TachyonBaseModel (Langchain base chat class, supports tool calls, no structured output)
- Must log message history incrementally to file
- Must validate results with SQL database connection

---

## Approach 1: LITE - Sequential Chunk Processing with Context Carryover

### Overview
Process the Ab-initio code sequentially in manageable chunks, maintaining context between chunks through a sliding window approach.

### Architecture
```
┌─────────────────┐
│  Load Files     │
└────────┬────────┘
         │
┌────────▼────────────────────────────────────┐
│  Intelligent Chunking (by graph sections)   │
│  - Split by .begin graph / .end graph       │
│  - Keep transformation blocks together       │
│  - Target ~300k chars per chunk             │
└────────┬────────────────────────────────────┘
         │
         │  For each chunk:
         │
┌────────▼─────────────────────┐
│  Extract Chunk Context       │
│  - Identify inputs/outputs   │
│  - Find dependencies         │
│  - Match with metadata files │
└────────┬─────────────────────┘
         │
┌────────▼─────────────────────┐
│  Convert Chunk to SQL        │
│  - Use LLM with full context │
│  - Reference prev chunk SQL  │
│  - Generate CTE or subquery  │
└────────┬─────────────────────┘
         │
┌────────▼─────────────────────┐
│  Accumulate SQL Fragments    │
│  - Store as CTEs             │
│  - Track dependencies        │
└────────┬─────────────────────┘
         │
         │ (Loop until all chunks processed)
         │
┌────────▼─────────────────────┐
│  Merge SQL Fragments         │
│  - Combine CTEs              │
│  - Add final SELECT          │
│  - Optimize query            │
└────────┬─────────────────────┘
         │
┌────────▼─────────────────────┐
│  Validate with SQL Tool      │
│  - Check syntax              │
│  - Test execution            │
│  - Verify row counts         │
└────────┬─────────────────────┘
         │
┌────────▼─────────────────────┐
│  Return Final SQL            │
└──────────────────────────────┘
```

### Agent States
```python
{
    "chunks": List[str],              # Ab-initio code chunks
    "current_index": int,             # Current chunk being processed
    "transformation_info": str,       # Transformation metadata
    "column_metadata": str,           # Column domain info
    "sql_fragments": List[dict],      # {chunk_id, sql, dependencies}
    "context_summary": str,           # Summary of previous chunks
    "final_sql": str,                 # Combined SQL query
    "validation_result": dict,        # Validation output
    "message_history": List[dict],    # For logging
    "errors": List[str]               # Error tracking
}
```

### LangGraph Nodes

1. **initialize_node**: Load files, parse and chunk Ab-initio code
2. **extract_chunk_context_node**: Analyze current chunk, identify I/O
3. **convert_chunk_node**: Convert chunk to SQL using TachyonBaseModel
4. **accumulate_node**: Store SQL fragment, update context summary
5. **merge_sql_node**: Combine all fragments into final query
6. **validate_sql_node**: Execute validation tool, check results
7. **refine_sql_node**: Fix issues if validation fails

### Tool: SQL Validator
```python
def sql_validation_tool(sql_query: str, connection_string: str) -> dict:
    """
    Returns:
    {
        "syntax_valid": bool,
        "execution_status": "success" | "error",
        "row_count": int,
        "sample_rows": List[dict],  # First 5 rows
        "execution_time_ms": float,
        "error_message": str | None,
        "column_names": List[str]
    }
    """
```

### Pros
- Simple to implement and debug
- Lower computational cost
- Works well for linear transformations
- Easier to trace errors

### Cons
- May lose global context between distant chunks
- Risk of inconsistent naming across chunks
- Limited ability to optimize across chunks
- May struggle with complex inter-chunk dependencies

### Best For
- Straightforward Ab-initio flows
- Well-structured code with clear sections
- Limited cross-graph references
- Quick prototyping

---

## Approach 2: STRONG - Hierarchical Map-Reduce with Validation Feedback Loop

### Overview
Two-phase approach: First extract complete structure and metadata (map), then convert intelligently based on dependencies (reduce), with continuous validation and refinement.

### Architecture
```
┌─────────────────────────────────┐
│  Load & Parse Files             │
└────────┬────────────────────────┘
         │
┌────────▼──────────────────────────────────┐
│  PHASE 1: STRUCTURE EXTRACTION (MAP)      │
│  ┌──────────────────────────────────┐    │
│  │ Chunk Ab-initio into sections    │    │
│  └──────────┬───────────────────────┘    │
│             │                             │
│  ┌──────────▼───────────────────────┐    │
│  │ For each chunk in parallel:      │    │
│  │ - Extract graph names             │    │
│  │ - Extract transformations         │    │
│  │ - Extract parameters              │    │
│  │ - Identify inputs/outputs         │    │
│  │ - Extract join conditions         │    │
│  └──────────┬───────────────────────┘    │
│             │                             │
│  ┌──────────▼───────────────────────┐    │
│  │ Build Global Metadata             │    │
│  │ - Dependency graph                │    │
│  │ - Data lineage map                │    │
│  │ - Transformation catalog          │    │
│  └──────────┬───────────────────────┘    │
└─────────────┼───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  PHASE 2: SQL GENERATION (REDUCE)       │
│  ┌──────────────────────────────────┐   │
│  │ Topological Sort (dependencies)  │   │
│  └──────────┬───────────────────────┘   │
│             │                            │
│  ┌──────────▼───────────────────────┐   │
│  │ For each transformation:          │   │
│  │ - Get context from metadata       │   │
│  │ - Match with transformation info  │   │
│  │ - Match with column metadata      │   │
│  │ - Convert to SQL CTE/subquery     │   │
│  │ - Validate incrementally          │   │
│  └──────────┬───────────────────────┘   │
│             │                            │
│  ┌──────────▼───────────────────────┐   │
│  │ Assemble Final SQL                │   │
│  │ - WITH cte1 AS (...),             │   │
│  │       cte2 AS (...),              │   │
│  │   SELECT * FROM final_cte         │   │
│  └──────────┬───────────────────────┘   │
└─────────────┼───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  PHASE 3: VALIDATION & REFINEMENT       │
│  ┌──────────────────────────────────┐   │
│  │ Execute SQL Validation Tool      │   │
│  │ - Syntax check                    │   │
│  │ - Execution test                  │   │
│  │ - Row count verification          │   │
│  │ - Column name validation          │   │
│  └──────────┬───────────────────────┘   │
│             │                            │
│       ┌─────▼──────┐                    │
│       │ Success?   │                    │
│       └─────┬──────┘                    │
│          No │   Yes                     │
│  ┌──────────▼───────────────────────┐   │
│  │ Error Analysis & Refinement      │   │
│  │ - Parse error message             │   │
│  │ - Identify problematic CTE        │   │
│  │ - Regenerate with error context   │   │
│  │ - Re-validate                     │   │
│  └──────────┬───────────────────────┘   │
│             │                            │
│             └────────────┐               │
│                          │               │
└──────────────────────────┼───────────────┘
                           │
                ┌──────────▼──────────┐
                │  Return Final SQL   │
                └─────────────────────┘
```

### Agent States
```python
{
    # Phase 1: Structure Extraction
    "raw_chunks": List[str],
    "chunk_metadata": List[dict],        # Extracted structure per chunk
    "global_dependency_graph": dict,     # {node: {depends_on: [...], outputs: [...]}}
    "transformation_catalog": dict,      # {trans_id: {...details...}}
    "data_lineage": dict,                # Track data flow

    # Phase 2: SQL Generation
    "conversion_order": List[str],       # Topologically sorted transformation IDs
    "sql_ctes": List[dict],              # {cte_name, sql, dependencies}
    "current_trans_index": int,

    # Context Files
    "transformation_info": str,
    "column_metadata": str,

    # Phase 3: Validation
    "final_sql": str,
    "validation_attempts": int,
    "validation_result": dict,
    "refinement_history": List[dict],

    # Logging
    "message_history": List[dict],
    "errors": List[str]
}
```

### LangGraph Nodes

**Phase 1 - Extraction:**
1. **parse_and_chunk_node**: Intelligent chunking by graph boundaries
2. **extract_metadata_node**: Extract structure from each chunk (can parallelize)
3. **build_dependency_graph_node**: Create global view of dependencies
4. **catalog_transformations_node**: Index all transformations

**Phase 2 - Conversion:**
5. **topological_sort_node**: Order transformations by dependencies
6. **convert_transformation_node**: Convert one transformation to SQL CTE
7. **validate_cte_node**: Incrementally validate each CTE
8. **assemble_sql_node**: Combine all CTEs into final query

**Phase 3 - Validation:**
9. **full_validation_node**: Execute complete SQL validation tool
10. **analyze_errors_node**: Parse validation errors
11. **refine_sql_node**: Fix identified issues
12. **final_check_node**: Verify final output

### Tool: Advanced SQL Validator
```python
def advanced_sql_validation_tool(
    sql_query: str,
    connection_string: str,
    validate_schema: bool = True,
    check_row_count: bool = True,
    sample_size: int = 10
) -> dict:
    """
    Returns:
    {
        "syntax_valid": bool,
        "execution_status": "success" | "error",
        "error_details": {
            "message": str,
            "line_number": int,
            "problematic_cte": str | None,
            "suggestion": str
        },
        "performance": {
            "execution_time_ms": float,
            "rows_affected": int,
            "query_plan": str  # EXPLAIN output
        },
        "data_validation": {
            "row_count": int,
            "null_counts": dict,  # {column: null_count}
            "sample_rows": List[dict],
            "column_names": List[str],
            "data_types": dict
        },
        "warnings": List[str]  # Performance or data quality warnings
    }
    """
```

### Advanced Features

1. **Intelligent Chunking**: Parse .begin/.end graph markers, keep transformations intact
2. **Parallel Metadata Extraction**: Process multiple chunks concurrently in Phase 1
3. **Dependency-Aware Conversion**: Convert in correct order, reference previous CTEs
4. **Incremental Validation**: Catch errors early at CTE level
5. **Feedback Loop**: Use validation errors to improve conversion
6. **Context Compression**: Summarize large metadata for repeated use

### Pros
- Handles complex dependencies correctly
- Better global optimization opportunities
- Incremental validation catches errors early
- Feedback loop improves accuracy
- More robust for large, complex code

### Cons
- More complex to implement
- Higher computational cost
- Longer initial processing time
- Requires sophisticated error handling

### Best For
- Complex Ab-initio flows with many dependencies
- Code with cross-graph references
- Production-grade conversion requirements
- Cases where accuracy is critical

---

## Comparison Matrix

| Aspect | Lite Approach | Strong Approach |
|--------|---------------|-----------------|
| **Complexity** | Low | High |
| **Processing Time** | Fast | Slower |
| **Accuracy** | Good for simple cases | Excellent for all cases |
| **Dependency Handling** | Sequential | Graph-based |
| **Error Recovery** | Basic | Advanced feedback loop |
| **Context Preservation** | Sliding window | Full global view |
| **Validation** | End-to-end only | Incremental + end-to-end |
| **Resource Usage** | Lower | Higher |
| **Debugging** | Easier | More complex |
| **Scalability** | Limited | High |

---

## Recommended Approach

**Start with Lite, upgrade to Strong if needed:**

1. Try Lite approach first for faster results
2. If validation shows issues with dependencies or context, switch to Strong
3. For production use or complex code, go directly with Strong approach

---

## Implementation Notes

### Message History Logging
```python
class IncrementalLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.message_count = 0

    def log_message(self, role: str, content: str):
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] Message #{self.message_count} - Role: {role}\n")
            f.write(f"{'-'*80}\n")
            f.write(content)
            f.write(f"\n{'='*80}\n")
            self.message_count += 1
```

### TachyonBaseModel Usage
```python
# Initialize
llm = TachyonBaseModel(model="claude", temperature=0.1)

# For tool calls
llm_with_tools = llm.bind_tools([sql_validation_tool])

# Invoke
response = llm_with_tools.invoke(messages)
```

### File Loading Strategy
- Load all 3 files at initialization
- For chunking: Use regex to find .begin/.end graph markers
- Target chunk size: 300k characters (leaving headroom for metadata)
- Always include relevant transformation_info and column_metadata with each chunk

---

## Success Metrics

1. **Correctness**: SQL produces expected results (validate with known test cases)
2. **Completeness**: All transformations converted (no missing logic)
3. **Performance**: SQL executes within acceptable time
4. **Maintainability**: Generated SQL is readable and follows best practices
5. **Validation**: Passes all SQL validator checks
