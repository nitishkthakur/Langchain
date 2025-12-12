# Parser + SQL Agent Integration Guide

## Combining the Parser with the Ab-initio to SQL Conversion Agent

This guide shows how to use the comprehensive Ab-initio parser with the SQL conversion agent for optimal results.

---

## Integration Strategy

### Approach 1: Pre-Parse Then Convert

Parse the Ab-initio code first to get structured JSON, then use that structure to guide SQL conversion.

```python
from ab_initio_claude_parser import parse_file_to_json, validate_parsed_graph
from abinitio_to_sql_agent import AbInitioToSQLAgent

# Step 1: Parse Ab-initio file
print("Parsing Ab-initio graph...")
graph_dict = parse_file_to_json("graphs/customer_etl.mp", "parsed_graph.json")

# Step 2: Validate for SQL conversion readiness
validation = validate_parsed_graph(graph_dict)
print(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")

if not validation['valid']:
    print("Errors found:")
    for error in validation['errors']:
        print(f"  - {error}")
    exit(1)

# Step 3: Create enhanced transformation metadata from parsed structure
transformation_info = create_transformation_info_from_parsed(graph_dict)
column_metadata = create_column_metadata_from_parsed(graph_dict)

# Step 4: Run SQL conversion agent with enhanced metadata
agent = AbInitioToSQLAgent(
    llm=llm,
    mode="strong",
    log_file="conversion_with_parser.log"
)

result = agent.convert(
    abinitio_file="graphs/customer_etl.mp",
    transformation_file="enhanced_transformations.txt",
    column_metadata_file="enhanced_columns.txt"
)

print(f"Generated SQL:\n{result['final_sql']}")
```

### Helper Functions

```python
import json

def create_transformation_info_from_parsed(graph_dict):
    """
    Generate enhanced transformation metadata from parsed graph.

    This creates a comprehensive transformation guide for the SQL agent.
    """
    transformations = []

    for component in graph_dict['components']:
        comp_type = component['component_type']
        comp_name = component['name']

        if comp_type == 'Reformat':
            transformations.append(f"""
Transformation: {comp_name}
Type: Reformat
Input Ports: {', '.join([p['name'] for p in component.get('input_ports', [])])}
Output Ports: {', '.join([p['name'] for p in component.get('output_ports', [])])}
Expression: {component.get('transform_expression', {}).get('expression', 'N/A')}
SQL Mapping: SELECT statement with field transformations
""")

        elif comp_type == 'Join':
            transformations.append(f"""
Transformation: {comp_name}
Type: Join
Join Type: {component.get('join_type', 'inner').upper()}
Join Keys: {', '.join(component.get('join_keys', []))}
Join Condition: {component.get('join_condition', 'N/A')}
SQL Mapping: {component.get('join_type', 'INNER').upper()} JOIN ON {' AND '.join([f'a.{k} = b.{k}' for k in component.get('join_keys', [])])}
""")

        elif comp_type in ['Rollup', 'Aggregate']:
            transformations.append(f"""
Transformation: {comp_name}
Type: Aggregation
Group By: {', '.join(component.get('group_by_keys', []))}
Aggregations: {json.dumps(component.get('aggregations', []), indent=2)}
SQL Mapping: GROUP BY {', '.join(component.get('group_by_keys', []))} with aggregate functions
""")

        elif comp_type == 'Filter by Expression':
            transformations.append(f"""
Transformation: {comp_name}
Type: Filter
Filter Expression: {component.get('filter_expression', 'N/A')}
SQL Mapping: WHERE {component.get('filter_expression', 'TRUE')}
""")

        elif comp_type == 'Sort':
            sort_spec = ', '.join([f"{s['key']} {s['order'].upper()}" for s in component.get('sort_keys', [])])
            transformations.append(f"""
Transformation: {comp_name}
Type: Sort
Sort Keys: {sort_spec}
SQL Mapping: ORDER BY {sort_spec}
""")

    # Add flow information
    transformations.append("\n\nData Flows:")
    for flow in graph_dict.get('flows', []):
        transformations.append(f"  {flow['source_component']}.{flow['source_port']} -> {flow['target_component']}.{flow['target_port']}")

    output_file = "enhanced_transformations.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(transformations))

    return output_file


def create_column_metadata_from_parsed(graph_dict):
    """
    Generate enhanced column metadata from parsed DML records.

    This creates detailed column information for the SQL agent.
    """
    columns_info = []

    columns_info.append("COLUMN METADATA FROM PARSED DML RECORDS\n")
    columns_info.append("="*80 + "\n")

    for dml_record in graph_dict.get('dml_records', []):
        columns_info.append(f"\nRecord: {dml_record['name']}")
        columns_info.append(f"Type: {dml_record['record_type']}")
        if dml_record.get('delimiter'):
            columns_info.append(f"Delimiter: {dml_record['delimiter']}")
        columns_info.append("\nFields:")

        for field in dml_record.get('fields', []):
            field_info = f"  - {field['name']}: {field['data_type']}"

            if field.get('length'):
                field_info += f"({field['length']})"
            elif field.get('precision'):
                field_info += f"({field['precision']},{field.get('scale', 0)})"

            if field.get('format'):
                field_info += f" format={field['format']}"

            if not field.get('nullable', True):
                field_info += " NOT NULL"

            if field.get('is_vector'):
                field_info += f" VECTOR[{field['vector_size']}]"

            # SQL type mapping
            sql_type = map_dml_to_sql_type(field)
            field_info += f" -> SQL: {sql_type}"

            columns_info.append(field_info)

    output_file = "enhanced_columns.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(columns_info))

    return output_file


def map_dml_to_sql_type(field):
    """Map DML field type to SQL type."""
    data_type = field['data_type'].lower()

    if data_type == 'string':
        length = field.get('length', 255)
        return f"VARCHAR({length})"

    elif data_type == 'decimal':
        precision = field.get('precision', 10)
        scale = field.get('scale', 0)
        return f"DECIMAL({precision},{scale})"

    elif data_type == 'integer':
        return "INTEGER"

    elif data_type == 'date':
        return "DATE"

    elif data_type == 'datetime':
        return "TIMESTAMP"

    elif data_type == 'boolean':
        return "BOOLEAN"

    elif data_type == 'vector':
        return "ARRAY" if field.get('is_vector') else "TEXT"

    else:
        return "TEXT"
```

---

## Approach 2: LLM Tool Integration

Let the LLM use the parser as a tool when it needs structured information.

```python
from ab_initio_claude_parser import create_abinitio_parser_tool
from abinitio_to_sql_agent import AbInitioToSQLAgent
from langchain.tools import StructuredTool

# Create parser tool
parser_function = create_abinitio_parser_tool()

parser_tool = StructuredTool.from_function(
    func=parser_function,
    name="abinitio_parser",
    description="""
    Parse Ab-initio ETL code into structured JSON format.

    Use this tool to:
    - Extract component information (types, configurations)
    - Parse DML record formats and field definitions
    - Get parameter values
    - Understand data flows between components

    Modes:
    - full: Parse complete graph structure
    - components: Parse only components
    - dml: Parse only DML records
    - parameters: Parse only parameters
    - flows: Parse only data flows

    Use this tool BEFORE attempting SQL conversion to get accurate structure.
    """
)

# Bind tool to LLM
llm_with_parser = llm.bind_tools([parser_tool])

# Use in agent
agent = AbInitioToSQLAgent(
    llm=llm_with_parser,
    mode="strong"
)

# The LLM can now call the parser tool as needed during conversion
result = agent.convert(
    abinitio_file="graphs/complex_etl.mp",
    transformation_file="transformations.txt",
    column_metadata_file="columns.txt"
)
```

---

## Approach 3: Hybrid - Parse Key Sections, LLM Converts

Parse the structure and let the LLM handle complex logic conversion.

```python
from ab_initio_claude_parser import AbInitioParser

parser = AbInitioParser()

# Read Ab-initio code
with open("graphs/customer_etl.mp", 'r') as f:
    abinitio_code = f.read()

# Parse structure
graph = parser.parse_content(abinitio_code)

# Strategy: Convert component-by-component using parsed structure
for component in graph.components:
    print(f"\nConverting: {component.name} ({component.component_type})")

    # Build context for LLM with parsed information
    if component.component_type == 'Reformat':
        context = f"""
Convert this Reformat component to SQL:

Component: {component.name}
Input Ports: {[p.name for p in component.input_ports]}
Output Ports: {[p.name for p in component.output_ports]}

Transform Expression:
{component.transform_expression.expression if component.transform_expression else 'N/A'}

Input Record Format:
{get_record_format_for_port(graph, component.input_ports[0]) if component.input_ports else 'N/A'}

Output Record Format:
{get_record_format_for_port(graph, component.output_ports[0]) if component.output_ports else 'N/A'}

Generate SQL SELECT statement.
"""

        response = llm.invoke([HumanMessage(content=context)])
        print(f"SQL: {response.content}")

    elif component.component_type == 'Join':
        context = f"""
Convert this Join component to SQL:

Component: {component.name}
Join Type: {component.join_type}
Join Keys: {component.join_keys}
Join Condition: {component.join_condition}

Input Ports:
{chr(10).join([f"  - {p.name}: {p.record_format}" for p in component.input_ports])}

Generate SQL JOIN clause.
"""

        response = llm.invoke([HumanMessage(content=context)])
        print(f"SQL: {response.content}")


def get_record_format_for_port(graph, port):
    """Get DML record format for a port."""
    if not port.record_format:
        return "Unknown"

    for dml_record in graph.dml_records:
        if dml_record.name == port.record_format:
            fields = [f"{f.name} ({f.data_type})" for f in dml_record.fields]
            return f"{dml_record.name}: {', '.join(fields)}"

    return port.record_format
```

---

## Approach 4: Intelligent Chunking with Parser

Use the parser to intelligently chunk based on component boundaries.

```python
from ab_initio_claude_parser import AbInitioParser

def intelligent_chunking_with_parser(abinitio_file, max_chunk_size=300000):
    """
    Use parser to create intelligent chunks based on component boundaries.
    """
    parser = AbInitioParser()

    with open(abinitio_file, 'r') as f:
        content = f.read()

    # Parse to get structure
    graph = parser.parse_content(content)

    # Build dependency graph
    dep_graph = build_dependency_graph(graph)

    # Topological sort to get processing order
    processing_order = topological_sort(dep_graph)

    # Create chunks based on dependencies
    chunks = []
    current_chunk = []
    current_size = 0

    for comp_name in processing_order:
        component = next(c for c in graph.components if c.name == comp_name)

        # Estimate size (rough approximation)
        comp_size = estimate_component_size(component, content)

        if current_size + comp_size > max_chunk_size and current_chunk:
            chunks.append({
                'components': current_chunk.copy(),
                'size': current_size
            })
            current_chunk = [comp_name]
            current_size = comp_size
        else:
            current_chunk.append(comp_name)
            current_size += comp_size

    if current_chunk:
        chunks.append({
            'components': current_chunk,
            'size': current_size
        })

    return chunks, graph


def build_dependency_graph(graph):
    """Build dependency graph from flows."""
    dep_graph = {c.name: [] for c in graph.components}

    for flow in graph.flows:
        target = flow.target_component
        source = flow.source_component
        if target in dep_graph:
            dep_graph[target].append(source)

    return dep_graph


def topological_sort(dep_graph):
    """Topological sort of components."""
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)
        for neighbor in dep_graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    for node in dep_graph:
        if node not in visited:
            dfs(node)

    return stack[::-1]


# Use intelligent chunking
chunks, graph = intelligent_chunking_with_parser("graphs/large_etl.mp")

print(f"Created {len(chunks)} intelligent chunks:")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk['components'])} components, ~{chunk['size']} chars")
    print(f"    Components: {', '.join(chunk['components'])}")
```

---

## Best Practices

### 1. Always Validate First

```python
graph_dict = parse_file_to_json("graph.mp")
validation = validate_parsed_graph(graph_dict)

if not validation['valid']:
    # Handle errors before proceeding
    fix_validation_errors(graph_dict, validation['errors'])
```

### 2. Use Parser for Structure, LLM for Logic

```python
# Parser extracts structure
components = parser._parse_components(code)
dml_records = parser._parse_dml_records(code)

# LLM handles complex transform logic
for component in components:
    if component.component_type == 'Reformat':
        # Use LLM to convert transform expression
        sql = llm_convert_transform(component.transform_expression)
```

### 3. Leverage Parsed Metadata

```python
# Use parsed DML for accurate SQL types
for dml_record in graph.dml_records:
    create_table_sql = generate_create_table(dml_record)

# Use parsed flows for JOIN order
join_order = determine_join_order_from_flows(graph.flows)

# Use parsed parameters for dynamic SQL
sql_with_params = substitute_parameters(sql_template, graph.parameters)
```

### 4. Component-Type Specific Handling

```python
def convert_component_to_sql(component, graph):
    """Convert based on component type using parsed info."""

    if component.component_type == 'Rollup':
        return generate_rollup_sql(
            group_by=component.group_by_keys,
            aggregations=component.aggregations,
            input_format=get_input_format(component, graph)
        )

    elif component.component_type == 'Join':
        return generate_join_sql(
            join_type=component.join_type,
            join_keys=component.join_keys,
            join_condition=component.join_condition,
            inputs=component.input_ports
        )

    # ... handle all component types
```

---

## Performance Optimization

### Use Parser for Large Files

```python
# Instead of sending entire 2M char file to LLM:
graph = parser.parse_file("large_graph.mp")

# Send only relevant chunks with structured context
for component in graph.components:
    # Much smaller, structured context for LLM
    component_context = create_component_context(component, graph)
    sql = llm.invoke(component_context)
```

### Cache Parsed Results

```python
import pickle

# Parse once, cache result
graph = parser.parse_file("graph.mp")
with open("parsed_cache.pkl", 'wb') as f:
    pickle.dump(graph.to_dict(), f)

# Reuse in multiple conversions
with open("parsed_cache.pkl", 'rb') as f:
    cached_graph = pickle.load(f)
```

---

## Error Handling

```python
try:
    # Parse
    graph = parser.parse_file("graph.mp")

    # Validate
    validation = validate_parsed_graph(graph.to_dict())
    if not validation['valid']:
        raise ValueError(f"Validation failed: {validation['errors']}")

    # Convert
    agent = AbInitioToSQLAgent(llm=llm)
    result = agent.convert(...)

    # Validate SQL
    if result['validation_result']['execution_status'] == 'error':
        raise RuntimeError(f"SQL validation failed: {result['validation_result']['error_details']}")

except FileNotFoundError as e:
    logger.error(f"Ab-initio file not found: {e}")
except ValueError as e:
    logger.error(f"Validation error: {e}")
except RuntimeError as e:
    logger.error(f"SQL execution error: {e}")
```

---

## Complete Integration Example

```python
from ab_initio_claude_parser import (
    parse_file_to_json,
    validate_parsed_graph,
    create_abinitio_parser_tool
)
from abinitio_to_sql_agent import AbInitioToSQLAgent

def convert_abinitio_to_sql_with_parser(
    abinitio_file: str,
    transformation_file: str,
    column_metadata_file: str,
    llm,
    use_parser_tool: bool = True
):
    """
    Complete workflow for Ab-initio to SQL conversion using parser.
    """

    # Step 1: Parse and validate
    print("Step 1: Parsing Ab-initio graph...")
    graph_dict = parse_file_to_json(abinitio_file, "parsed_graph.json")

    print("Step 2: Validating...")
    validation = validate_parsed_graph(graph_dict)

    if not validation['valid']:
        print("VALIDATION FAILED:")
        for error in validation['errors']:
            print(f"  ERROR: {error}")
        return None

    print(f"✓ Validation passed")
    print(f"  Components: {validation['component_count']}")
    print(f"  Flows: {validation['flow_count']}")
    print(f"  DML Records: {validation['dml_record_count']}")

    # Step 2: Enhance metadata with parsed information
    print("\nStep 3: Enhancing metadata with parsed structure...")
    enhanced_transform = create_transformation_info_from_parsed(graph_dict)
    enhanced_columns = create_column_metadata_from_parsed(graph_dict)

    # Step 3: Setup LLM with optional parser tool
    if use_parser_tool:
        print("\nStep 4: Creating parser tool for LLM...")
        parser_tool = create_abinitio_parser_tool()
        llm_with_tools = llm.bind_tools([parser_tool])
    else:
        llm_with_tools = llm

    # Step 4: Run conversion
    print("\nStep 5: Running SQL conversion agent...")
    agent = AbInitioToSQLAgent(
        llm=llm_with_tools,
        mode="strong",
        log_file="conversion_with_parser.log"
    )

    result = agent.convert(
        abinitio_file=abinitio_file,
        transformation_file=enhanced_transform,
        column_metadata_file=enhanced_columns
    )

    # Step 5: Report results
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)

    print(f"\nStatistics:")
    print(f"  Messages: {result['message_count']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  CTEs: {result['num_ctes']}")

    if result['validation_result'].get('syntax_valid'):
        print(f"  ✓ SQL syntax is valid")
    else:
        print(f"  ✗ SQL syntax errors detected")

    if result['errors']:
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")

    # Save results
    with open("final_query.sql", 'w') as f:
        f.write(result['final_sql'])

    print(f"\nGenerated SQL saved to: final_query.sql")
    print(f"Parsed structure saved to: parsed_graph.json")
    print(f"Conversion log saved to: conversion_with_parser.log")

    return result


# Run the complete workflow
if __name__ == "__main__":
    from your_module import TachyonBaseModel

    llm = TachyonBaseModel(model="claude", temperature=0.1)

    result = convert_abinitio_to_sql_with_parser(
        abinitio_file="graphs/customer_etl.mp",
        transformation_file="transformations.txt",
        column_metadata_file="columns.txt",
        llm=llm,
        use_parser_tool=True
    )
```

---

## Summary

The parser + SQL agent integration provides:

1. **Structure Extraction**: Parse before conversion for accurate structure
2. **Enhanced Metadata**: Generate detailed transformation/column info
3. **LLM Tool**: Let LLM call parser when needed
4. **Intelligent Chunking**: Component-aware chunking
5. **Validation**: Ensure completeness before SQL generation
6. **Type Mapping**: Accurate DML → SQL type conversion

Choose the approach that best fits your use case!
