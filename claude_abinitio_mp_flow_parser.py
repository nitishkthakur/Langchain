"""
Ab Initio .mp File Parser

This parser extracts and analyzes Ab Initio graph (.mp) files to:
1. Identify separate components/flows
2. Determine flow dependencies (which flows output to which)
3. Resolve in/out parameter mappings for each component
4. Generate comprehensive flow analysis and visualization

Author: Generated for Ab Initio Graph Analysis
"""

import re
import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import argparse


@dataclass
class ComponentParameter:
    """Represents an input or output parameter of a component"""
    param_type: str  # 'in' or 'out'
    param_index: int  # 0, 1, 2, etc.
    flow_name: str  # Name of the flow connected to this parameter


@dataclass
class Component:
    """Represents an Ab Initio component in the graph"""
    name: str
    component_type: str
    inputs: List[ComponentParameter] = field(default_factory=list)
    outputs: List[ComponentParameter] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def get_input_mapping(self) -> Dict[str, str]:
        """Returns mapping of input parameters to flow names"""
        return {f"in{p.param_index}": p.flow_name for p in self.inputs}

    def get_output_mapping(self) -> Dict[str, str]:
        """Returns mapping of output parameters to flow names"""
        return {f"out{p.param_index}": p.flow_name for p in self.outputs}


@dataclass
class Flow:
    """Represents a data flow between components"""
    flow_name: str
    source_component: str
    target_component: str
    source_output_port: int = 0
    target_input_port: int = 0


@dataclass
class GraphMetadata:
    """Represents graph-level metadata"""
    name: str = ""
    version: str = ""
    description: str = ""


class AbInitioMPParser:
    """Parser for Ab Initio .mp graph files"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.components: Dict[str, Component] = {}
        self.flows: List[Flow] = []
        self.graph_metadata = GraphMetadata()
        self.flow_to_components: Dict[str, Tuple[str, str]] = {}  # flow_name -> (source, target)

    def parse(self) -> None:
        """Main parsing method"""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Parse graph metadata
        self._parse_graph_metadata(content)

        # Parse all components
        self._parse_components(content)

        # Parse flow definitions
        self._parse_flows(content)

        # Resolve component parameters based on flows
        self._resolve_component_parameters()

    def _parse_graph_metadata(self, content: str) -> None:
        """Extract graph-level metadata"""
        # Extract graph name
        name_match = re.search(r'name:\s*([^\n]+)', content)
        if name_match:
            self.graph_metadata.name = name_match.group(1).strip()

        # Extract version
        version_match = re.search(r'version:\s*([^\n]+)', content)
        if version_match:
            self.graph_metadata.version = version_match.group(1).strip()

        # Extract description
        desc_match = re.search(r'description:\s*([^\n]+)', content)
        if desc_match:
            self.graph_metadata.description = desc_match.group(1).strip()

    def _parse_components(self, content: str) -> None:
        """Extract all components from the graph"""
        # Find all component blocks
        component_pattern = r'begin component\s+(.*?)\s+end component'
        component_blocks = re.findall(component_pattern, content, re.DOTALL)

        for block in component_blocks:
            component = self._parse_component_block(block)
            if component:
                self.components[component.name] = component

    def _parse_component_block(self, block: str) -> Component:
        """Parse a single component block"""
        # Extract component type
        type_match = re.search(r'type:\s*([^\n]+)', block)
        component_type = type_match.group(1).strip() if type_match else "unknown"

        # Extract component name
        name_match = re.search(r'name:\s*([^\n]+)', block)
        component_name = name_match.group(1).strip() if name_match else "unnamed"

        # Extract metadata
        metadata = {}
        for key in ['file_path', 'layout', 'filter_condition', 'transform',
                    'join_type', 'join_key', 'group_by', 'sort_keys', 'rollup_key', 'lookup_key']:
            match = re.search(rf'{key}:\s*([^\n]+)', block)
            if match:
                metadata[key] = match.group(1).strip()

        # Create component
        component = Component(
            name=component_name,
            component_type=component_type,
            metadata=metadata
        )

        # Extract parameters section
        params_match = re.search(r'begin parameters\s+(.*?)\s+end parameters', block, re.DOTALL)
        if params_match:
            params_content = params_match.group(1)
            self._parse_parameters(component, params_content)

        return component

    def _parse_parameters(self, component: Component, params_content: str) -> None:
        """Parse parameters section of a component"""
        # Find all in/out parameter definitions
        param_pattern = r'(in|out)(\d+):\s*([^\n]+)'
        params = re.findall(param_pattern, params_content)

        for param_type, param_index, flow_name in params:
            flow_name = flow_name.strip()
            param_index = int(param_index)

            param = ComponentParameter(
                param_type=param_type,
                param_index=param_index,
                flow_name=flow_name
            )

            if param_type == 'in':
                component.inputs.append(param)
            else:
                component.outputs.append(param)

    def _parse_flows(self, content: str) -> None:
        """Extract flow definitions"""
        # Find flows section
        flows_match = re.search(r'begin flows\s+(.*?)\s+end flows', content, re.DOTALL)
        if not flows_match:
            return

        flows_content = flows_match.group(1)

        # Parse individual flow definitions
        # Format: flow: flow_name from source_component to target_component
        flow_pattern = r'flow:\s*(\S+)\s+from\s+(\S+)\s+to\s+(\S+)'
        flow_matches = re.findall(flow_pattern, flows_content)

        for flow_name, source_comp, target_comp in flow_matches:
            flow = Flow(
                flow_name=flow_name,
                source_component=source_comp,
                target_component=target_comp
            )
            self.flows.append(flow)
            self.flow_to_components[flow_name] = (source_comp, target_comp)

    def _resolve_component_parameters(self) -> None:
        """Resolve and validate component parameters against flows"""
        # This method ensures consistency between component parameters and flow definitions
        for flow in self.flows:
            # Update source component's output port
            if flow.source_component in self.components:
                source_comp = self.components[flow.source_component]
                for output in source_comp.outputs:
                    if output.flow_name == flow.flow_name:
                        flow.source_output_port = output.param_index

            # Update target component's input port
            if flow.target_component in self.components:
                target_comp = self.components[flow.target_component]
                for input_param in target_comp.inputs:
                    if input_param.flow_name == flow.flow_name:
                        flow.target_input_port = input_param.param_index

    def get_component_dependencies(self) -> Dict[str, List[str]]:
        """
        Get component dependencies - which components output to which components
        Returns: Dict mapping component_name -> list of downstream component names
        """
        dependencies = defaultdict(list)

        for flow in self.flows:
            dependencies[flow.source_component].append(flow.target_component)

        return dict(dependencies)

    def get_flow_lineage(self, flow_name: str) -> List[str]:
        """
        Trace the lineage of a specific flow through the graph
        Returns: List of component names in order
        """
        if flow_name not in self.flow_to_components:
            return []

        source, target = self.flow_to_components[flow_name]
        return [source, target]

    def get_component_flows(self, component_name: str) -> Dict[str, List[str]]:
        """
        Get all flows connected to a specific component
        Returns: Dict with 'inputs' and 'outputs' lists
        """
        result = {
            'inputs': [],
            'outputs': []
        }

        if component_name not in self.components:
            return result

        component = self.components[component_name]

        # Get input flows
        result['inputs'] = [inp.flow_name for inp in component.inputs]

        # Get output flows
        result['outputs'] = [out.flow_name for out in component.outputs]

        return result

    def get_graph_topology(self) -> Dict[str, any]:
        """
        Get complete graph topology with all relationships
        """
        topology = {
            'metadata': {
                'name': self.graph_metadata.name,
                'version': self.graph_metadata.version,
                'description': self.graph_metadata.description
            },
            'components': {},
            'flows': [],
            'dependencies': self.get_component_dependencies()
        }

        # Add component details
        for comp_name, component in self.components.items():
            topology['components'][comp_name] = {
                'type': component.component_type,
                'input_mapping': component.get_input_mapping(),
                'output_mapping': component.get_output_mapping(),
                'metadata': component.metadata
            }

        # Add flow details
        for flow in self.flows:
            topology['flows'].append({
                'name': flow.flow_name,
                'source': flow.source_component,
                'target': flow.target_component,
                'source_port': f'out{flow.source_output_port}',
                'target_port': f'in{flow.target_input_port}'
            })

        return topology

    def generate_report(self) -> str:
        """Generate a comprehensive text report of the graph analysis"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Ab Initio Graph Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Graph metadata
        report_lines.append("GRAPH METADATA:")
        report_lines.append(f"  Name: {self.graph_metadata.name}")
        report_lines.append(f"  Version: {self.graph_metadata.version}")
        report_lines.append(f"  Description: {self.graph_metadata.description}")
        report_lines.append("")

        # Components summary
        report_lines.append(f"TOTAL COMPONENTS: {len(self.components)}")
        report_lines.append("")

        # Component details with I/O mappings
        report_lines.append("COMPONENT DETAILS:")
        report_lines.append("-" * 80)
        for comp_name, component in sorted(self.components.items()):
            report_lines.append(f"\nComponent: {comp_name}")
            report_lines.append(f"  Type: {component.component_type}")

            if component.metadata:
                report_lines.append("  Metadata:")
                for key, value in component.metadata.items():
                    report_lines.append(f"    {key}: {value}")

            # Input mapping
            input_mapping = component.get_input_mapping()
            if input_mapping:
                report_lines.append("  Input Mapping:")
                for port, flow in sorted(input_mapping.items()):
                    report_lines.append(f"    {port}: {flow}")

            # Output mapping
            output_mapping = component.get_output_mapping()
            if output_mapping:
                report_lines.append("  Output Mapping:")
                for port, flow in sorted(output_mapping.items()):
                    report_lines.append(f"    {port}: {flow}")

        # Flow details
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"TOTAL FLOWS: {len(self.flows)}")
        report_lines.append("=" * 80)
        report_lines.append("")
        for flow in self.flows:
            report_lines.append(f"Flow: {flow.flow_name}")
            report_lines.append(f"  {flow.source_component}[out{flow.source_output_port}] "
                              f"--> {flow.target_component}[in{flow.target_input_port}]")

        # Dependencies
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("COMPONENT DEPENDENCIES:")
        report_lines.append("=" * 80)
        dependencies = self.get_component_dependencies()
        for comp_name, downstream in sorted(dependencies.items()):
            report_lines.append(f"\n{comp_name} outputs to:")
            for target in downstream:
                report_lines.append(f"  -> {target}")

        # Flow lineage for each flow
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("FLOW LINEAGE:")
        report_lines.append("=" * 80)
        for flow in self.flows:
            lineage = self.get_flow_lineage(flow.flow_name)
            report_lines.append(f"\n{flow.flow_name}: {' -> '.join(lineage)}")

        return "\n".join(report_lines)

    def export_to_json(self, output_file: str) -> None:
        """Export complete graph topology to JSON"""
        topology = self.get_graph_topology()
        with open(output_file, 'w') as f:
            json.dump(topology, f, indent=2)

    def print_summary(self) -> None:
        """Print a quick summary of the graph"""
        print(f"\nGraph: {self.graph_metadata.name}")
        print(f"Components: {len(self.components)}")
        print(f"Flows: {len(self.flows)}")
        print(f"\nComponent Types:")
        type_counts = defaultdict(int)
        for component in self.components.values():
            type_counts[component.component_type] += 1
        for comp_type, count in sorted(type_counts.items()):
            print(f"  {comp_type}: {count}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Parse Ab Initio .mp graph files and extract flow information'
    )
    parser.add_argument('mp_file', help='Path to the .mp file to parse')
    parser.add_argument('-o', '--output', help='Output file for detailed report (text)')
    parser.add_argument('-j', '--json', help='Output file for JSON export')
    parser.add_argument('-s', '--summary', action='store_true',
                       help='Print summary to console')

    args = parser.parse_args()

    # Create parser and parse the file
    mp_parser = AbInitioMPParser(args.mp_file)
    print(f"Parsing {args.mp_file}...")
    mp_parser.parse()
    print("Parsing complete!")

    # Print summary if requested
    if args.summary:
        mp_parser.print_summary()

    # Generate text report
    if args.output:
        print(f"\nGenerating detailed report to {args.output}...")
        report = mp_parser.generate_report()
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")

    # Export to JSON
    if args.json:
        print(f"\nExporting graph topology to {args.json}...")
        mp_parser.export_to_json(args.json)
        print(f"JSON export saved to {args.json}")

    # If no output specified, print to console
    if not args.output and not args.json and not args.summary:
        print("\n" + mp_parser.generate_report())


if __name__ == "__main__":
    main()
