"""
Ab-initio MP File Parser - The Most Comprehensive Parser

This parser extracts ALL critical components from Ab-initio .mp graph files
and converts them to structured JSON format for SQL conversion and LLM processing.

Research Sources:
- Ab Initio components, transforms, and graph structures
- DML (Data Manipulation Language) formats
- XFR (Transform) expressions
- MFS (Multi-File System) configurations
- Metadata and flow definitions

Author: Claude Code
Date: 2025-12-12
Version: 1.0 - Ultimate Edition
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS FOR AB-INITIO COMPONENTS
# ============================================================================

class ComponentType(Enum):
    """All Ab-initio component types identified from research."""

    # Input/Output Components
    INPUT_FILE = "Input File"
    OUTPUT_FILE = "Output File"
    LOOKUP_FILE = "Lookup File"
    INPUT_TABLE = "Input Table"
    OUTPUT_TABLE = "Output Table"

    # Transform Components
    REFORMAT = "Reformat"
    ROLLUP = "Rollup"
    JOIN = "Join"
    NORMALIZE = "Normalize"
    COMBINE = "Combine"
    SCAN = "Scan"
    FILTER_BY_EXPRESSION = "Filter by Expression"
    DEDUP_SORTED = "Dedup Sorted"
    AGGREGATE = "Aggregate"  # Deprecated, but may appear
    MULTI_REFORMAT = "Multi Reformat"

    # Partition Components
    PARTITION_BY_KEY = "Partition by Key"
    PARTITION_BY_EXPRESSION = "Partition by Expression"
    PARTITION_BY_ROUND_ROBIN = "Partition by Round Robin"
    PARTITION_BY_PERCENTAGE = "Partition by Percentage"
    PARTITION_BY_LOAD_BALANCE = "Partition by Load Balance"
    BROADCAST = "Broadcast"
    REPLICATE = "Replicate"

    # De-partition Components
    GATHER = "Gather"
    MERGE = "Merge"
    INTERLEAVE = "Interleave"
    CONCATENATE = "Concatenate"

    # Sort Components
    SORT = "Sort"
    SORT_WITHIN_GROUPS = "Sort Within Groups"

    # Database Components
    DB_INPUT = "Database Input"
    DB_OUTPUT = "Database Output"
    DB_LOAD = "Database Load"

    # Special Components
    SUBGRAPH = "Subgraph"
    CUSTOM_COMPONENT = "Custom Component"
    CHECKPOINT = "Checkpoint"
    CONTINUOUS_SUBSCRIBE = "Continuous Subscribe"
    CONTINUOUS_PUBLISH = "Continuous Publish"

    # Utility Components
    TRASH = "Trash"
    FTP_GET = "FTP Get"
    FTP_PUT = "FTP Put"

    UNKNOWN = "Unknown"


class DataType(Enum):
    """DML data types."""
    STRING = "string"
    DECIMAL = "decimal"
    INTEGER = "integer"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    BOOLEAN = "boolean"
    VECTOR = "vector"
    RECORD = "record"
    RAW = "raw"


class JoinType(Enum):
    """Join types in Ab-initio."""
    INNER = "inner"
    LEFT_OUTER = "left outer"
    RIGHT_OUTER = "right outer"
    FULL_OUTER = "full outer"
    CROSS = "cross"


class PortType(Enum):
    """Port types."""
    INPUT = "input"
    OUTPUT = "output"
    LOOKUP = "lookup"
    REJECT = "reject"
    UNUSED = "unused"


# ============================================================================
# DATA CLASSES FOR STRUCTURED REPRESENTATION
# ============================================================================

@dataclass
class DMLField:
    """Represents a field in a DML record format."""
    name: str
    data_type: str
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    delimiter: Optional[str] = None
    format: Optional[str] = None
    nullable: bool = True
    default_value: Optional[str] = None
    vector_size: Optional[int] = None
    is_vector: bool = False
    nested_fields: Optional[List['DMLField']] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable
        }
        if self.length is not None:
            result["length"] = self.length
        if self.precision is not None:
            result["precision"] = self.precision
        if self.scale is not None:
            result["scale"] = self.scale
        if self.delimiter:
            result["delimiter"] = self.delimiter
        if self.format:
            result["format"] = self.format
        if self.default_value:
            result["default_value"] = self.default_value
        if self.is_vector:
            result["is_vector"] = True
            result["vector_size"] = self.vector_size
        if self.nested_fields:
            result["nested_fields"] = [f.to_dict() for f in self.nested_fields]
        return result


@dataclass
class DMLRecord:
    """Represents a DML record format definition."""
    name: str
    fields: List[DMLField]
    record_type: str = "delimited"  # delimited, fixed, binary, etc.
    delimiter: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "record_type": self.record_type,
            "delimiter": self.delimiter,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields]
        }


@dataclass
class Port:
    """Represents a component port."""
    name: str
    port_type: str  # PortType
    record_format: Optional[str] = None
    dml_definition: Optional[DMLRecord] = None
    connected_to: Optional[str] = None
    partition_type: Optional[str] = None
    partition_key: Optional[List[str]] = None
    is_partitioned: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "port_type": self.port_type,
            "is_partitioned": self.is_partitioned
        }
        if self.record_format:
            result["record_format"] = self.record_format
        if self.dml_definition:
            result["dml_definition"] = self.dml_definition.to_dict()
        if self.connected_to:
            result["connected_to"] = self.connected_to
        if self.partition_type:
            result["partition_type"] = self.partition_type
        if self.partition_key:
            result["partition_key"] = self.partition_key
        return result


@dataclass
class TransformExpression:
    """Represents a transform expression (XFR)."""
    name: str
    input_ports: List[str]
    output_ports: List[str]
    expression: str
    language: str = "dml"  # dml, python, etc.
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "input_ports": self.input_ports,
            "output_ports": self.output_ports,
            "expression": self.expression,
            "language": self.language,
            "dependencies": self.dependencies
        }


@dataclass
class Component:
    """Represents an Ab-initio component."""
    id: str
    name: str
    component_type: str
    description: Optional[str] = None

    # Ports
    input_ports: List[Port] = field(default_factory=list)
    output_ports: List[Port] = field(default_factory=list)
    lookup_ports: List[Port] = field(default_factory=list)
    reject_ports: List[Port] = field(default_factory=list)

    # Transform-specific
    transform_expression: Optional[TransformExpression] = None

    # Join-specific
    join_type: Optional[str] = None
    join_condition: Optional[str] = None
    join_keys: Optional[List[str]] = None

    # Rollup/Aggregate-specific
    group_by_keys: Optional[List[str]] = None
    aggregations: Optional[List[Dict[str, str]]] = None

    # Filter-specific
    filter_expression: Optional[str] = None
    rejection_threshold: Optional[int] = None

    # Sort-specific
    sort_keys: Optional[List[Dict[str, str]]] = None  # [{key: field, order: asc/desc}]

    # Partition-specific
    partition_type: Optional[str] = None
    partition_key: Optional[List[str]] = None
    partition_expression: Optional[str] = None
    num_partitions: Optional[int] = None

    # Dedup-specific
    dedup_keys: Optional[List[str]] = None
    keep_option: Optional[str] = None  # first, last, unique-only

    # Normalize/Combine-specific
    vector_field: Optional[str] = None
    max_occurrences: Optional[int] = None

    # File/Table-specific
    file_path: Optional[str] = None
    table_name: Optional[str] = None
    database_connection: Optional[str] = None

    # Layout and execution
    layout: Optional[str] = None
    parallelism: Optional[int] = None

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "component_type": self.component_type
        }

        if self.description:
            result["description"] = self.description

        if self.input_ports:
            result["input_ports"] = [p.to_dict() for p in self.input_ports]
        if self.output_ports:
            result["output_ports"] = [p.to_dict() for p in self.output_ports]
        if self.lookup_ports:
            result["lookup_ports"] = [p.to_dict() for p in self.lookup_ports]
        if self.reject_ports:
            result["reject_ports"] = [p.to_dict() for p in self.reject_ports]

        if self.transform_expression:
            result["transform_expression"] = self.transform_expression.to_dict()

        # Add type-specific fields
        optional_fields = [
            'join_type', 'join_condition', 'join_keys',
            'group_by_keys', 'aggregations',
            'filter_expression', 'rejection_threshold',
            'sort_keys',
            'partition_type', 'partition_key', 'partition_expression', 'num_partitions',
            'dedup_keys', 'keep_option',
            'vector_field', 'max_occurrences',
            'file_path', 'table_name', 'database_connection',
            'layout', 'parallelism'
        ]

        for field_name in optional_fields:
            value = getattr(self, field_name, None)
            if value is not None:
                result[field_name] = value

        if self.parameters:
            result["parameters"] = self.parameters
        if self.depends_on:
            result["depends_on"] = self.depends_on

        return result


@dataclass
class Flow:
    """Represents a data flow between components."""
    id: str
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    record_format: Optional[str] = None
    is_partitioned: bool = False
    partition_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "source_component": self.source_component,
            "source_port": self.source_port,
            "target_component": self.target_component,
            "target_port": self.target_port,
            "is_partitioned": self.is_partitioned
        }
        if self.record_format:
            result["record_format"] = self.record_format
        if self.partition_info:
            result["partition_info"] = self.partition_info
        return result


@dataclass
class Parameter:
    """Represents a graph parameter."""
    name: str
    value: Any
    data_type: Optional[str] = None
    description: Optional[str] = None
    is_runtime: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "data_type": self.data_type,
            "description": self.description,
            "is_runtime": self.is_runtime
        }


@dataclass
class Phase:
    """Represents a graph phase."""
    id: str
    name: str
    components: List[str]
    description: Optional[str] = None
    is_checkpoint: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "components": self.components,
            "description": self.description,
            "is_checkpoint": self.is_checkpoint
        }


@dataclass
class Graph:
    """Represents a complete Ab-initio graph."""
    name: str
    description: Optional[str] = None

    # Main structures
    components: List[Component] = field(default_factory=list)
    flows: List[Flow] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    phases: List[Phase] = field(default_factory=list)
    dml_records: List[DMLRecord] = field(default_factory=list)

    # Metadata
    version: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None

    # Graph-level configurations
    default_layout: Optional[str] = None
    mfs_configuration: Optional[Dict[str, Any]] = None

    # Dependencies
    subgraphs: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "components": [c.to_dict() for c in self.components],
            "flows": [f.to_dict() for f in self.flows],
            "parameters": [p.to_dict() for p in self.parameters],
            "phases": [p.to_dict() for p in self.phases],
            "dml_records": [d.to_dict() for d in self.dml_records],
            "default_layout": self.default_layout,
            "mfs_configuration": self.mfs_configuration,
            "subgraphs": self.subgraphs,
            "external_dependencies": self.external_dependencies
        }


# ============================================================================
# AB-INITIO PARSER CLASS
# ============================================================================

class AbInitioParser:
    """
    Comprehensive Ab-initio MP file parser.

    Parses all critical components needed for SQL conversion:
    - Graph structure and metadata
    - All component types (transform, partition, join, etc.)
    - DML record formats
    - Transform expressions (XFR)
    - Data flows and connections
    - Parameters and phases
    - Layout and parallelism configurations
    """

    def __init__(self):
        """Initialize the parser."""
        self.component_id_counter = 0
        self.flow_id_counter = 0
        self.phase_id_counter = 0

    def parse_file(self, file_path: str) -> Graph:
        """
        Parse an Ab-initio .mp file.

        Args:
            file_path: Path to the .mp file

        Returns:
            Graph object containing all parsed information
        """
        logger.info(f"Parsing Ab-initio file: {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return self.parse_content(content)

    def parse_content(self, content: str) -> Graph:
        """
        Parse Ab-initio content from string.

        Args:
            content: Ab-initio code as string

        Returns:
            Graph object containing all parsed information
        """
        logger.info("Parsing Ab-initio content...")

        # Extract graph name
        graph_name = self._extract_graph_name(content)
        logger.info(f"Graph name: {graph_name}")

        # Initialize graph
        graph = Graph(name=graph_name)

        # Parse metadata
        graph.description = self._extract_description(content)
        graph.version = self._extract_version(content)

        # Parse parameters
        graph.parameters = self._parse_parameters(content)
        logger.info(f"Found {len(graph.parameters)} parameters")

        # Parse DML records
        graph.dml_records = self._parse_dml_records(content)
        logger.info(f"Found {len(graph.dml_records)} DML records")

        # Parse phases
        graph.phases = self._parse_phases(content)
        logger.info(f"Found {len(graph.phases)} phases")

        # Parse components
        graph.components = self._parse_components(content)
        logger.info(f"Found {len(graph.components)} components")

        # Parse flows
        graph.flows = self._parse_flows(content, graph.components)
        logger.info(f"Found {len(graph.flows)} flows")

        # Parse subgraphs
        graph.subgraphs = self._parse_subgraphs(content)

        # Parse MFS configuration
        graph.mfs_configuration = self._parse_mfs_config(content)

        # Build dependencies
        self._build_dependencies(graph)

        logger.info("Parsing complete!")
        return graph

    # ========================================================================
    # GRAPH-LEVEL PARSING
    # ========================================================================

    def _extract_graph_name(self, content: str) -> str:
        """Extract graph name from content."""
        patterns = [
            r'\.begin\s+graph\s+(\w+)',
            r'graph\s+name\s*=\s*"([^"]+)"',
            r'graph\s+name\s*=\s*\'([^\']+)\'',
            r'#\s*Graph:\s*(\w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return "unknown_graph"

    def _extract_description(self, content: str) -> Optional[str]:
        """Extract graph description."""
        patterns = [
            r'description\s*:\s*"([^"]+)"',
            r'description\s*=\s*"([^"]+)"',
            r'#\s*Description:\s*(.+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_version(self, content: str) -> Optional[str]:
        """Extract graph version."""
        patterns = [
            r'version\s*:\s*"([^"]+)"',
            r'version\s*=\s*"([^"]+)"',
            r'#\s*Version:\s*(.+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        return None

    # ========================================================================
    # PARAMETER PARSING
    # ========================================================================

    def _parse_parameters(self, content: str) -> List[Parameter]:
        """Parse all parameters from the graph."""
        parameters = []

        # Pattern 1: .parameter declarations
        param_pattern = r'\.parameter\s+(\w+)\s*=\s*(.+?)(?:;|$)'
        for match in re.finditer(param_pattern, content, re.MULTILINE):
            name = match.group(1)
            value_str = match.group(2).strip()

            # Try to infer type and parse value
            value, data_type = self._parse_parameter_value(value_str)

            parameters.append(Parameter(
                name=name,
                value=value,
                data_type=data_type,
                is_runtime=False
            ))

        # Pattern 2: Runtime parameters
        runtime_pattern = r'runtime\s+parameter\s+(\w+)\s*(?::\s*(\w+))?'
        for match in re.finditer(runtime_pattern, content, re.IGNORECASE):
            name = match.group(1)
            data_type = match.group(2) if match.group(2) else None

            parameters.append(Parameter(
                name=name,
                value=None,  # Runtime - value not known at parse time
                data_type=data_type,
                is_runtime=True
            ))

        # Pattern 3: Variable assignments (xxparameter style)
        xx_param_pattern = r'xx(\w+)\s*=\s*(.+?)(?:;|$)'
        for match in re.finditer(xx_param_pattern, content, re.MULTILINE):
            name = match.group(1)
            value_str = match.group(2).strip()
            value, data_type = self._parse_parameter_value(value_str)

            parameters.append(Parameter(
                name=name,
                value=value,
                data_type=data_type,
                is_runtime=False
            ))

        return parameters

    def _parse_parameter_value(self, value_str: str) -> Tuple[Any, Optional[str]]:
        """Parse parameter value and infer type."""
        value_str = value_str.strip().rstrip(';')

        # Remove quotes
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1], "string"

        # Try integer
        try:
            return int(value_str), "integer"
        except ValueError:
            pass

        # Try decimal
        try:
            return float(value_str), "decimal"
        except ValueError:
            pass

        # Boolean
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true', "boolean"

        # Default to string
        return value_str, "string"

    # ========================================================================
    # DML RECORD PARSING
    # ========================================================================

    def _parse_dml_records(self, content: str) -> List[DMLRecord]:
        """Parse all DML record format definitions."""
        records = []

        # Pattern: record ... end
        record_pattern = r'record\s+(?:(\w+)\s+)?(?:\{([^}]+)\}\s*)?(.*?)end(?:\s*;)?'

        for match in re.finditer(record_pattern, content, re.DOTALL | re.IGNORECASE):
            record_name = match.group(1) if match.group(1) else f"record_{len(records)}"
            options = match.group(2) if match.group(2) else ""
            fields_text = match.group(3)

            # Parse fields
            fields = self._parse_dml_fields(fields_text)

            # Determine delimiter
            delimiter = None
            if 'delimited' in options.lower():
                delim_match = re.search(r'delimited\s*\(?\s*["\']([^"\']+)["\']', options)
                if delim_match:
                    delimiter = delim_match.group(1)

            record_type = "delimited" if delimiter or "delimited" in options.lower() else "fixed"

            records.append(DMLRecord(
                name=record_name,
                fields=fields,
                record_type=record_type,
                delimiter=delimiter
            ))

        return records

    def _parse_dml_fields(self, fields_text: str) -> List[DMLField]:
        """Parse DML field definitions."""
        fields = []

        # Split by semicolon but handle nested records
        field_lines = []
        current_line = ""
        depth = 0

        for char in fields_text:
            current_line += char
            if char == '{' or (char == 'r' and 'record' in current_line[-10:]):
                depth += 1
            elif char == '}' or (char == 'e' and 'end' in current_line[-10:]):
                depth -= 1
            elif char == ';' and depth == 0:
                field_lines.append(current_line.strip())
                current_line = ""

        if current_line.strip():
            field_lines.append(current_line.strip())

        # Parse each field
        for line in field_lines:
            line = line.strip().rstrip(';')
            if not line or line.startswith('//') or line.startswith('#'):
                continue

            field = self._parse_single_dml_field(line)
            if field:
                fields.append(field)

        return fields

    def _parse_single_dml_field(self, line: str) -> Optional[DMLField]:
        """Parse a single DML field definition."""
        line = line.strip()

        # Handle comments
        line = re.sub(r'//.*$', '', line)
        line = re.sub(r'#.*$', '', line)
        line = line.strip()

        if not line:
            return None

        # Pattern: type(params) field_name [delimiter] [= default]
        # Examples:
        # - string(10) name;
        # - decimal(10,2) amount;
        # - date("YYYYMMDD") birth_date;
        # - string field_name;
        # - vector items[10];

        # Check for vector
        vector_match = re.match(r'(\w+)\s+(\w+)\[(\d+)\]', line)
        if vector_match:
            data_type = vector_match.group(1)
            field_name = vector_match.group(2)
            vector_size = int(vector_match.group(3))

            return DMLField(
                name=field_name,
                data_type=data_type,
                is_vector=True,
                vector_size=vector_size
            )

        # Standard field pattern
        field_pattern = r'(\w+)\s*(?:\(([^)]+)\))?\s+(\w+)\s*(?:\[([^\]]+)\])?\s*(?:=\s*(.+))?'
        match = re.match(field_pattern, line)

        if not match:
            return None

        data_type = match.group(1)
        params = match.group(2)
        field_name = match.group(3)
        delimiter_spec = match.group(4)
        default_value = match.group(5)

        # Parse parameters (length, precision, scale, format)
        length = None
        precision = None
        scale = None
        format_str = None
        delimiter = None

        if params:
            params = params.strip()

            # Check for format string (in quotes)
            format_match = re.match(r'["\']([^"\']+)["\']', params)
            if format_match:
                format_str = format_match.group(1)
            else:
                # Numeric parameters
                if ',' in params:
                    parts = params.split(',')
                    precision = int(parts[0].strip())
                    scale = int(parts[1].strip()) if len(parts) > 1 else None
                else:
                    try:
                        length = int(params)
                    except ValueError:
                        format_str = params

        if delimiter_spec:
            delimiter = delimiter_spec.strip('"\'')

        return DMLField(
            name=field_name,
            data_type=data_type,
            length=length,
            precision=precision,
            scale=scale,
            format=format_str,
            delimiter=delimiter,
            default_value=default_value.strip() if default_value else None
        )

    # ========================================================================
    # PHASE PARSING
    # ========================================================================

    def _parse_phases(self, content: str) -> List[Phase]:
        """Parse graph phases."""
        phases = []

        # Pattern: .begin phase ... .end phase
        phase_pattern = r'\.begin\s+phase\s+(\w+)(.*?)\.end\s+phase'

        for match in re.finditer(phase_pattern, content, re.DOTALL | re.IGNORECASE):
            phase_name = match.group(1)
            phase_content = match.group(2)

            # Extract components in this phase
            component_names = re.findall(r'component\s+(\w+)', phase_content, re.IGNORECASE)

            # Check if checkpoint
            is_checkpoint = 'checkpoint' in phase_content.lower()

            phases.append(Phase(
                id=f"phase_{self.phase_id_counter}",
                name=phase_name,
                components=component_names,
                is_checkpoint=is_checkpoint
            ))
            self.phase_id_counter += 1

        return phases

    # ========================================================================
    # COMPONENT PARSING
    # ========================================================================

    def _parse_components(self, content: str) -> List[Component]:
        """Parse all components from the graph."""
        components = []

        # Try to find component definitions
        # Pattern variations depending on Ab-initio syntax

        # Pattern 1: Component with type and name
        comp_pattern1 = r'(?:component|comp)\s+(\w+)\s*:\s*(\w+(?:\s+\w+)*)'

        for match in re.finditer(comp_pattern1, content, re.IGNORECASE):
            comp_name = match.group(1)
            comp_type_str = match.group(2)

            # Identify component type
            comp_type = self._identify_component_type(comp_type_str)

            component = Component(
                id=f"comp_{self.component_id_counter}",
                name=comp_name,
                component_type=comp_type.value
            )

            # Extract component-specific details based on type
            self._parse_component_details(component, content, comp_name)

            components.append(component)
            self.component_id_counter += 1

        # If no components found with pattern 1, try alternative patterns
        if not components:
            components = self._parse_components_alternative(content)

        return components

    def _identify_component_type(self, type_str: str) -> ComponentType:
        """Identify component type from string."""
        type_str_lower = type_str.lower()

        # Map common variations to ComponentType
        type_mapping = {
            'input file': ComponentType.INPUT_FILE,
            'input': ComponentType.INPUT_FILE,
            'output file': ComponentType.OUTPUT_FILE,
            'output': ComponentType.OUTPUT_FILE,
            'lookup': ComponentType.LOOKUP_FILE,
            'lookup file': ComponentType.LOOKUP_FILE,
            'reformat': ComponentType.REFORMAT,
            'rollup': ComponentType.ROLLUP,
            'join': ComponentType.JOIN,
            'normalize': ComponentType.NORMALIZE,
            'combine': ComponentType.COMBINE,
            'scan': ComponentType.SCAN,
            'filter': ComponentType.FILTER_BY_EXPRESSION,
            'filter by expression': ComponentType.FILTER_BY_EXPRESSION,
            'dedup': ComponentType.DEDUP_SORTED,
            'dedup sorted': ComponentType.DEDUP_SORTED,
            'aggregate': ComponentType.AGGREGATE,
            'partition by key': ComponentType.PARTITION_BY_KEY,
            'partition by expression': ComponentType.PARTITION_BY_EXPRESSION,
            'partition by round robin': ComponentType.PARTITION_BY_ROUND_ROBIN,
            'broadcast': ComponentType.BROADCAST,
            'replicate': ComponentType.REPLICATE,
            'gather': ComponentType.GATHER,
            'merge': ComponentType.MERGE,
            'interleave': ComponentType.INTERLEAVE,
            'concatenate': ComponentType.CONCATENATE,
            'sort': ComponentType.SORT,
            'sort within groups': ComponentType.SORT_WITHIN_GROUPS,
            'db input': ComponentType.DB_INPUT,
            'db output': ComponentType.DB_OUTPUT,
            'subgraph': ComponentType.SUBGRAPH,
        }

        for key, comp_type in type_mapping.items():
            if key in type_str_lower:
                return comp_type

        return ComponentType.UNKNOWN

    def _parse_component_details(self, component: Component, content: str, comp_name: str):
        """Parse detailed information for a specific component."""

        # Find component block in content
        comp_block_pattern = rf'{comp_name}\s*:\s*[^{{]*\{{(.*?)\}}'
        match = re.search(comp_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            return

        comp_content = match.group(1)

        # Parse based on component type
        comp_type = component.component_type

        if 'Reformat' in comp_type:
            self._parse_reformat_details(component, comp_content)
        elif 'Join' in comp_type:
            self._parse_join_details(component, comp_content)
        elif 'Rollup' in comp_type or 'Aggregate' in comp_type:
            self._parse_rollup_details(component, comp_content)
        elif 'Filter' in comp_type:
            self._parse_filter_details(component, comp_content)
        elif 'Sort' in comp_type:
            self._parse_sort_details(component, comp_content)
        elif 'Dedup' in comp_type:
            self._parse_dedup_details(component, comp_content)
        elif 'Partition' in comp_type:
            self._parse_partition_details(component, comp_content)
        elif 'Input' in comp_type or 'Output' in comp_type:
            self._parse_io_details(component, comp_content)

        # Parse ports (common for all)
        self._parse_component_ports(component, comp_content)

    def _parse_reformat_details(self, component: Component, content: str):
        """Parse Reformat component details."""
        # Look for transform expression
        xfr_pattern = r'transform\s*=\s*(.+?)(?:;|$)'
        match = re.search(xfr_pattern, content, re.DOTALL)

        if match:
            xfr_expr = match.group(1).strip()
            component.transform_expression = TransformExpression(
                name=f"{component.name}_transform",
                input_ports=["in"],
                output_ports=["out"],
                expression=xfr_expr
            )

    def _parse_join_details(self, component: Component, content: str):
        """Parse Join component details."""
        # Join type
        join_type_pattern = r'join\s+type\s*=\s*"([^"]+)"'
        match = re.search(join_type_pattern, content, re.IGNORECASE)
        if match:
            component.join_type = match.group(1).lower()

        # Join keys
        key_pattern = r'key\s*=\s*\(([^)]+)\)'
        match = re.search(key_pattern, content)
        if match:
            keys = [k.strip() for k in match.group(1).split(',')]
            component.join_keys = keys

        # Join condition
        condition_pattern = r'condition\s*=\s*"([^"]+)"'
        match = re.search(condition_pattern, content)
        if match:
            component.join_condition = match.group(1)

    def _parse_rollup_details(self, component: Component, content: str):
        """Parse Rollup/Aggregate component details."""
        # Group by keys
        key_pattern = r'key\s*=\s*\(([^)]+)\)'
        match = re.search(key_pattern, content)
        if match:
            keys = [k.strip() for k in match.group(1).split(',')]
            component.group_by_keys = keys

        # Aggregations
        agg_pattern = r'(sum|count|avg|min|max|first|last)\s*\(([^)]+)\)'
        aggregations = []
        for match in re.finditer(agg_pattern, content, re.IGNORECASE):
            aggregations.append({
                "function": match.group(1).lower(),
                "field": match.group(2).strip()
            })
        if aggregations:
            component.aggregations = aggregations

    def _parse_filter_details(self, component: Component, content: str):
        """Parse Filter component details."""
        filter_pattern = r'filter\s*=\s*"([^"]+)"'
        match = re.search(filter_pattern, content)
        if match:
            component.filter_expression = match.group(1)

        # Rejection threshold
        reject_pattern = r'rejection\s+threshold\s*=\s*(\d+)'
        match = re.search(reject_pattern, content, re.IGNORECASE)
        if match:
            component.rejection_threshold = int(match.group(1))

    def _parse_sort_details(self, component: Component, content: str):
        """Parse Sort component details."""
        # Sort keys with order
        key_pattern = r'key\s*=\s*\(([^)]+)\)'
        match = re.search(key_pattern, content)
        if match:
            keys_str = match.group(1)
            sort_keys = []
            for key_spec in keys_str.split(','):
                key_spec = key_spec.strip()
                if 'desc' in key_spec.lower():
                    key_name = re.sub(r'\s*desc\s*$', '', key_spec, flags=re.IGNORECASE).strip()
                    sort_keys.append({"key": key_name, "order": "desc"})
                else:
                    key_name = re.sub(r'\s*asc\s*$', '', key_spec, flags=re.IGNORECASE).strip()
                    sort_keys.append({"key": key_name, "order": "asc"})
            component.sort_keys = sort_keys

    def _parse_dedup_details(self, component: Component, content: str):
        """Parse Dedup component details."""
        # Dedup keys
        key_pattern = r'key\s*=\s*\(([^)]+)\)'
        match = re.search(key_pattern, content)
        if match:
            keys = [k.strip() for k in match.group(1).split(',')]
            component.dedup_keys = keys

        # Keep option
        keep_pattern = r'keep\s*=\s*"?(\w+(?:-\w+)?)"?'
        match = re.search(keep_pattern, content, re.IGNORECASE)
        if match:
            component.keep_option = match.group(1).lower()

    def _parse_partition_details(self, component: Component, content: str):
        """Parse Partition component details."""
        # Partition key
        key_pattern = r'key\s*=\s*\(([^)]+)\)'
        match = re.search(key_pattern, content)
        if match:
            keys = [k.strip() for k in match.group(1).split(',')]
            component.partition_key = keys

        # Number of partitions
        num_pattern = r'partitions?\s*=\s*(\d+)'
        match = re.search(num_pattern, content, re.IGNORECASE)
        if match:
            component.num_partitions = int(match.group(1))

        # Partition expression
        expr_pattern = r'expression\s*=\s*"([^"]+)"'
        match = re.search(expr_pattern, content)
        if match:
            component.partition_expression = match.group(1)

    def _parse_io_details(self, component: Component, content: str):
        """Parse Input/Output component details."""
        # File path
        file_pattern = r'file\s*=\s*"([^"]+)"'
        match = re.search(file_pattern, content)
        if match:
            component.file_path = match.group(1)

        # Table name
        table_pattern = r'table\s*=\s*"([^"]+)"'
        match = re.search(table_pattern, content)
        if match:
            component.table_name = match.group(1)

        # Database connection
        db_pattern = r'database\s*=\s*"([^"]+)"'
        match = re.search(db_pattern, content)
        if match:
            component.database_connection = match.group(1)

    def _parse_component_ports(self, component: Component, content: str):
        """Parse component ports."""
        # Input ports
        in_port_pattern = r'in(?:put)?\s+port\s+(\w+)'
        for match in re.finditer(in_port_pattern, content, re.IGNORECASE):
            port_name = match.group(1)
            component.input_ports.append(Port(
                name=port_name,
                port_type=PortType.INPUT.value
            ))

        # Output ports
        out_port_pattern = r'out(?:put)?\s+port\s+(\w+)'
        for match in re.finditer(out_port_pattern, content, re.IGNORECASE):
            port_name = match.group(1)
            component.output_ports.append(Port(
                name=port_name,
                port_type=PortType.OUTPUT.value
            ))

        # Lookup ports
        lookup_port_pattern = r'lookup\s+port\s+(\w+)'
        for match in re.finditer(lookup_port_pattern, content, re.IGNORECASE):
            port_name = match.group(1)
            component.lookup_ports.append(Port(
                name=port_name,
                port_type=PortType.LOOKUP.value
            ))

    def _parse_components_alternative(self, content: str) -> List[Component]:
        """Alternative component parsing strategy."""
        # This is a fallback - create placeholder components based on mentions
        components = []

        # Look for common component mentions
        component_keywords = [
            'input', 'output', 'reformat', 'join', 'rollup', 'filter',
            'sort', 'dedup', 'partition', 'gather', 'merge'
        ]

        for keyword in component_keywords:
            pattern = rf'\b{keyword}\b\s+(\w+)'
            for match in re.finditer(pattern, content, re.IGNORECASE):
                comp_name = match.group(1)
                comp_type = self._identify_component_type(keyword)

                components.append(Component(
                    id=f"comp_{self.component_id_counter}",
                    name=comp_name,
                    component_type=comp_type.value
                ))
                self.component_id_counter += 1

        return components

    # ========================================================================
    # FLOW PARSING
    # ========================================================================

    def _parse_flows(self, content: str, components: List[Component]) -> List[Flow]:
        """Parse data flows between components."""
        flows = []

        # Pattern: comp1.out -> comp2.in
        flow_pattern = r'(\w+)\.(\w+)\s*-+>\s*(\w+)\.(\w+)'

        for match in re.finditer(flow_pattern, content):
            source_comp = match.group(1)
            source_port = match.group(2)
            target_comp = match.group(3)
            target_port = match.group(4)

            flows.append(Flow(
                id=f"flow_{self.flow_id_counter}",
                source_component=source_comp,
                source_port=source_port,
                target_component=target_comp,
                target_port=target_port
            ))
            self.flow_id_counter += 1

        # Alternative pattern: flow from ... to ...
        flow_pattern2 = r'flow\s+from\s+(\w+)(?:\.(\w+))?\s+to\s+(\w+)(?:\.(\w+))?'

        for match in re.finditer(flow_pattern2, content, re.IGNORECASE):
            source_comp = match.group(1)
            source_port = match.group(2) if match.group(2) else "out"
            target_comp = match.group(3)
            target_port = match.group(4) if match.group(4) else "in"

            flows.append(Flow(
                id=f"flow_{self.flow_id_counter}",
                source_component=source_comp,
                source_port=source_port,
                target_component=target_comp,
                target_port=target_port
            ))
            self.flow_id_counter += 1

        return flows

    # ========================================================================
    # SUBGRAPH AND DEPENDENCIES
    # ========================================================================

    def _parse_subgraphs(self, content: str) -> List[str]:
        """Parse subgraph references."""
        subgraphs = []

        pattern = r'subgraph\s+(\w+)'
        for match in re.finditer(pattern, content, re.IGNORECASE):
            subgraphs.append(match.group(1))

        return subgraphs

    def _parse_mfs_config(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse MFS (Multi-File System) configuration."""
        mfs_config = {}

        # Look for MFS settings
        mfs_pattern = r'AI_MFS(?:_(\w+))?_HOME\s*=\s*"([^"]+)"'
        for match in re.finditer(mfs_pattern, content):
            variant = match.group(1) if match.group(1) else "default"
            path = match.group(2)
            mfs_config[variant] = path

        # Parallelism degree
        parallel_pattern = r'(\d+)-way\s+(?:parallel|partition)'
        match = re.search(parallel_pattern, content, re.IGNORECASE)
        if match:
            mfs_config["parallelism"] = int(match.group(1))

        return mfs_config if mfs_config else None

    def _build_dependencies(self, graph: Graph):
        """Build component dependencies based on flows."""
        # Create a mapping of component names to components
        comp_map = {c.name: c for c in graph.components}

        # Build dependencies from flows
        for flow in graph.flows:
            source_comp_name = flow.source_component
            target_comp_name = flow.target_component

            if target_comp_name in comp_map:
                if source_comp_name not in comp_map[target_comp_name].depends_on:
                    comp_map[target_comp_name].depends_on.append(source_comp_name)


# ============================================================================
# LLM TOOL INTERFACE
# ============================================================================

def create_abinitio_parser_tool():
    """
    Creates a tool function that can be used with LLMs.

    This allows the LLM to invoke the parser on specific sections
    or the entire Ab-initio code.

    Returns:
        A function that can be bound to an LLM as a tool.
    """

    def abinitio_parser_tool(
        code: str,
        parse_mode: str = "full",
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse Ab-initio code and return structured JSON.

        Args:
            code: Ab-initio code to parse
            parse_mode: Parsing mode - "full", "components", "dml", "parameters", "flows"
            section: Optional specific section to parse

        Returns:
            Dictionary with parsed structure in JSON format
        """
        parser = AbInitioParser()

        try:
            if parse_mode == "full":
                # Parse entire graph
                graph = parser.parse_content(code)
                return {
                    "status": "success",
                    "parsed_graph": graph.to_dict()
                }

            elif parse_mode == "components":
                # Parse only components
                components = parser._parse_components(code)
                return {
                    "status": "success",
                    "components": [c.to_dict() for c in components]
                }

            elif parse_mode == "dml":
                # Parse only DML records
                dml_records = parser._parse_dml_records(code)
                return {
                    "status": "success",
                    "dml_records": [d.to_dict() for d in dml_records]
                }

            elif parse_mode == "parameters":
                # Parse only parameters
                parameters = parser._parse_parameters(code)
                return {
                    "status": "success",
                    "parameters": [p.to_dict() for p in parameters]
                }

            elif parse_mode == "flows":
                # Parse flows (need components first)
                components = parser._parse_components(code)
                flows = parser._parse_flows(code, components)
                return {
                    "status": "success",
                    "flows": [f.to_dict() for f in flows]
                }

            else:
                return {
                    "status": "error",
                    "message": f"Unknown parse_mode: {parse_mode}"
                }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }

    return abinitio_parser_tool


# ============================================================================
# MAIN EXECUTION AND UTILITIES
# ============================================================================

def parse_file_to_json(file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse an Ab-initio file and optionally save to JSON.

    Args:
        file_path: Path to Ab-initio .mp file
        output_path: Optional path to save JSON output

    Returns:
        Parsed graph as dictionary
    """
    parser = AbInitioParser()
    graph = parser.parse_file(file_path)
    graph_dict = graph.to_dict()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved parsed graph to: {output_path}")

    return graph_dict


def parse_string_to_json(code: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse Ab-initio code string and optionally save to JSON.

    Args:
        code: Ab-initio code as string
        output_path: Optional path to save JSON output

    Returns:
        Parsed graph as dictionary
    """
    parser = AbInitioParser()
    graph = parser.parse_content(code)
    graph_dict = graph.to_dict()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved parsed graph to: {output_path}")

    return graph_dict


def validate_parsed_graph(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a parsed graph for completeness and SQL conversion readiness.

    Args:
        graph_dict: Parsed graph dictionary

    Returns:
        Validation report with warnings and errors
    """
    warnings = []
    errors = []

    # Check for components
    if not graph_dict.get('components'):
        errors.append("No components found in graph")

    # Check for flows
    if not graph_dict.get('flows'):
        warnings.append("No flows found - components may be disconnected")

    # Check each component
    for comp in graph_dict.get('components', []):
        comp_name = comp.get('name', 'unknown')

        # Check for input/output ports
        if not comp.get('input_ports') and 'Input' not in comp.get('component_type', ''):
            warnings.append(f"Component '{comp_name}' has no input ports")

        if not comp.get('output_ports') and 'Output' not in comp.get('component_type', ''):
            warnings.append(f"Component '{comp_name}' has no output ports")

        # Check transform components have expressions
        if 'Reformat' in comp.get('component_type', ''):
            if not comp.get('transform_expression'):
                warnings.append(f"Reformat component '{comp_name}' has no transform expression")

        # Check join components
        if 'Join' in comp.get('component_type', ''):
            if not comp.get('join_keys') and not comp.get('join_condition'):
                warnings.append(f"Join component '{comp_name}' has no join keys or condition")

    # Check DML records
    if not graph_dict.get('dml_records'):
        warnings.append("No DML record formats found")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "component_count": len(graph_dict.get('components', [])),
        "flow_count": len(graph_dict.get('flows', [])),
        "dml_record_count": len(graph_dict.get('dml_records', []))
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the Ab-initio parser."""

    print("="*80)
    print("AB-INITIO COMPREHENSIVE PARSER")
    print("The Most Extensive Ab-initio MP File Parser Ever")
    print("="*80)

    # Example 1: Parse from file
    example_file = "path/to/your/abinitio_graph.mp"

    if Path(example_file).exists():
        print(f"\nParsing file: {example_file}")
        graph_dict = parse_file_to_json(example_file, "parsed_graph.json")

        print(f"\nParsed Graph: {graph_dict['name']}")
        print(f"Components: {len(graph_dict['components'])}")
        print(f"Flows: {len(graph_dict['flows'])}")
        print(f"Parameters: {len(graph_dict['parameters'])}")
        print(f"DML Records: {len(graph_dict['dml_records'])}")

        # Validate
        validation = validate_parsed_graph(graph_dict)
        print(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
        if validation['warnings']:
            print(f"Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings'][:5]:  # Show first 5
                print(f"  - {warning}")
        if validation['errors']:
            print(f"Errors: {len(validation['errors'])}")
            for error in validation['errors']:
                print(f"  - {error}")

    # Example 2: Parse from string
    sample_code = """
    .begin graph sample_etl

    .parameter input_file = "${DATA_DIR}/input.dat"
    .parameter output_file = "${DATA_DIR}/output.dat"

    record input_record {
        string(10) customer_id;
        string(50) customer_name;
        decimal(10,2) amount;
        date("YYYYMMDD") transaction_date;
    }

    component read_input: Input File {
        file = ${input_file}
        record_format = input_record
    }

    component filter_data: Filter by Expression {
        filter = "amount > 100.00"
    }

    component write_output: Output File {
        file = ${output_file}
    }

    flow from read_input to filter_data
    flow from filter_data to write_output

    .end graph
    """

    print("\n" + "="*80)
    print("Parsing sample code...")
    graph_dict = parse_string_to_json(sample_code, "sample_parsed.json")

    print(f"\nParsed Graph: {graph_dict['name']}")
    print(f"Components: {len(graph_dict['components'])}")
    print(f"Flows: {len(graph_dict['flows'])}")

    # Example 3: Use as LLM tool
    print("\n" + "="*80)
    print("Creating LLM tool interface...")

    parser_tool = create_abinitio_parser_tool()

    # Simulate LLM calling the tool
    result = parser_tool(
        code=sample_code,
        parse_mode="components"
    )

    print(f"Tool result status: {result['status']}")
    if result['status'] == 'success':
        print(f"Components parsed: {len(result['components'])}")
        for comp in result['components']:
            print(f"  - {comp['name']} ({comp['component_type']})")

    print("\n" + "="*80)
    print("Parser demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
