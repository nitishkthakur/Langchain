"""
Focused Ab Initio .mp flow parser.

The goal here is to reliably extract data-flow wiring from Ab Initio graph
exports (.mp) without needing a full semantic parser. The parser:
1) Reads a .mp file (or string) and strips comments.
2) Captures port-to-flow bindings declared inside component blocks, e.g.
      component read_input: Input File {
          out0: FLOW_READ
      }
      component filter_data: Filter by Expression {
          in0: FLOW_READ
          out0: FLOW_FILTERED
      }
3) Captures inline connectors such as:
      read_input.out0 -> filter_data.in0
      flow FLOW_FILTERED: filter_data.out0 -> write_output.in0
      flow from filter_data to write_output
4) Builds a flow map showing, for each flow name, which component ports
   publish to it (sources) and which consume from it (targets), plus the
   concrete edges that connect the two sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import re


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=True)
class PortRef:
    component: str
    port: str

    def to_dict(self) -> Dict[str, str]:
        return {"component": self.component, "port": self.port}


@dataclass
class FlowEdge:
    flow_name: Optional[str]
    source: PortRef
    target: PortRef
    raw: str

    def with_flow(self, flow_name: str) -> "FlowEdge":
        """Return a copy with flow_name set (avoids mutating during merge)."""
        return FlowEdge(flow_name=flow_name, source=self.source, target=self.target, raw=self.raw)

    def to_dict(self) -> Dict[str, str]:
        return {
            "flow_name": self.flow_name,
            "source_component": self.source.component,
            "source_port": self.source.port,
            "target_component": self.target.component,
            "target_port": self.target.port,
            "raw": self.raw,
        }


@dataclass
class FlowGroup:
    flow_name: str
    sources: List[PortRef] = field(default_factory=list)
    targets: List[PortRef] = field(default_factory=list)
    edges: List[FlowEdge] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_source(self, ref: PortRef):
        if ref not in self.sources:
            self.sources.append(ref)

    def add_target(self, ref: PortRef):
        if ref not in self.targets:
            self.targets.append(ref)

    def add_edge(self, edge: FlowEdge):
        # Deduplicate edges by the tuple of components/ports
        key = (edge.source.component, edge.source.port, edge.target.component, edge.target.port)
        if key not in {(e.source.component, e.source.port, e.target.component, e.target.port) for e in self.edges}:
            self.edges.append(edge)

    def to_dict(self) -> Dict:
        return {
            "flow_name": self.flow_name,
            "sources": [s.to_dict() for s in self.sources],
            "targets": [t.to_dict() for t in self.targets],
            "edges": [e.to_dict() for e in self.edges],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class AbInitioFlowParser:
    """Lightweight flow extractor for Ab Initio .mp graphs."""

    # Component declarations often look like:
    #   component read_input: Input File {
    # or
    #   read_input: Input Table {
    _component_start = re.compile(r"^\s*(?:component\s+)?(?P<name>[\w$.-]+)\s*:\s*[^\\n{]*\{?", re.IGNORECASE)
    # Port bindings inside a component:
    #   out0: FLOW_NAME
    #   in1 = FLOW_X
    _port_binding = re.compile(
        r"^\s*(?P<port>(?:in|out|reject|lookup)\w*)\s*[:=]\s*(?P<flow>[A-Za-z0-9_.-]+)", re.IGNORECASE
    )

    # Inline connectors across the graph.
    _named_flow = re.compile(
        r"flow\s+(?P<flow>[A-Za-z0-9_.-]+)\s*[:=]\s*"
        r"(?P<src_comp>[A-Za-z0-9_$-]+)(?:\.(?P<src_port>[A-Za-z0-9_$-]+))?\s*-+>\s*"
        r"(?P<tgt_comp>[A-Za-z0-9_$-]+)(?:\.(?P<tgt_port>[A-Za-z0-9_$-]+))?",
        re.IGNORECASE,
    )
    _flow_from = re.compile(
        r"flow\s+from\s+(?P<src_comp>[A-Za-z0-9_$-]+)(?:\.(?P<src_port>[A-Za-z0-9_$-]+))?"
        r"\s+to\s+(?P<tgt_comp>[A-Za-z0-9_$-]+)(?:\.(?P<tgt_port>[A-Za-z0-9_$-]+))?",
        re.IGNORECASE,
    )
    _connector = re.compile(
        r"(?P<src_comp>[A-Za-z0-9_$-]+)\.(?P<src_port>[A-Za-z0-9_$-]+)\s*-+>\s*"
        r"(?P<tgt_comp>[A-Za-z0-9_$-]+)\.(?P<tgt_port>[A-Za-z0-9_$-]+)"
    )

    def parse_file(self, file_path: Path | str) -> Dict:
        """Parse a .mp file from disk."""
        path = Path(file_path)
        content = path.read_text(errors="ignore")
        return self.parse(content, source=str(path))

    def parse(self, content: str, source: Optional[str] = None) -> Dict:
        """Parse .mp content and return a flow mapping."""
        cleaned = self._strip_comments(content)
        components = self._extract_component_ports(cleaned)
        explicit_edges = self._extract_inline_edges(cleaned)
        flow_groups = self._link_by_flow_name(components)
        self._merge_edges(flow_groups, explicit_edges)

        return {
            "source": source,
            "components": components,
            "flows": [fg.to_dict() for fg in flow_groups.values()],
            "explicit_edges": [edge.to_dict() for edge in explicit_edges],
        }

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _strip_comments(content: str) -> str:
        """Remove /* ... */, // ... and # ... comments."""
        without_block = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        cleaned_lines = []
        for line in without_block.splitlines():
            line = re.sub(r"//.*$", "", line)
            line = re.sub(r"#.*$", "", line)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _extract_component_ports(self, content: str) -> Dict[str, Dict]:
        """Grab port-to-flow bindings declared inside component blocks."""
        components: Dict[str, Dict] = {}
        current: Optional[str] = None
        brace_depth = 0

        for line in content.splitlines():
            if current is None:
                start = self._component_start.match(line)
                if start:
                    current = start.group("name")
                    components.setdefault(current, {"inputs": {}, "outputs": {}, "other_ports": {}, "lines": []})
                    brace_depth = line.count("{") - line.count("}")
                continue

            brace_depth += line.count("{") - line.count("}")
            if current:
                components[current]["lines"].append(line.strip())

            binding = self._port_binding.match(line)
            if current and binding:
                port = binding.group("port")
                flow_name = binding.group("flow")
                lower_port = port.lower()
                if lower_port.startswith("in"):
                    components[current]["inputs"][port] = flow_name
                elif lower_port.startswith("out") or lower_port.startswith("reject"):
                    components[current]["outputs"][port] = flow_name
                else:
                    components[current]["other_ports"][port] = flow_name

            if brace_depth <= 0 and "}" in line:
                current = None
                brace_depth = 0

        return components

    def _extract_inline_edges(self, content: str) -> List[FlowEdge]:
        """Find direct connectors in the graph text."""
        edges: List[FlowEdge] = []
        seen_spans = set()

        def _add(match, flow_name: Optional[str]):
            span = match.span()
            if span in seen_spans:
                return
            seen_spans.add(span)
            source = PortRef(match.group("src_comp"), match.group("src_port") or "out")
            target = PortRef(match.group("tgt_comp"), match.group("tgt_port") or "in")
            edges.append(FlowEdge(flow_name=flow_name, source=source, target=target, raw=match.group(0).strip()))

        for match in self._named_flow.finditer(content):
            _add(match, match.group("flow"))
        for match in self._flow_from.finditer(content):
            _add(match, None)
        for match in self._connector.finditer(content):
            _add(match, None)

        return edges

    def _link_by_flow_name(self, components: Dict[str, Dict]) -> Dict[str, FlowGroup]:
        """Connect sources and targets that share the same flow label."""
        flow_groups: Dict[str, FlowGroup] = {}

        for comp_name, ports in components.items():
            for port, flow_name in ports.get("outputs", {}).items():
                group = flow_groups.setdefault(flow_name, FlowGroup(flow_name))
                group.add_source(PortRef(comp_name, port))
            for port, flow_name in ports.get("inputs", {}).items():
                group = flow_groups.setdefault(flow_name, FlowGroup(flow_name))
                group.add_target(PortRef(comp_name, port))

        for flow_name, group in flow_groups.items():
            # Build edges via cartesian product of sources x targets
            for src in group.sources:
                for tgt in group.targets:
                    group.add_edge(
                        FlowEdge(
                            flow_name=flow_name,
                            source=src,
                            target=tgt,
                            raw="linked by shared flow label",
                        )
                    )
            if not group.sources or not group.targets:
                note = "Flow has only sources" if group.sources else "Flow has only targets"
                group.notes.append(note)

        return flow_groups

    def _merge_edges(self, flow_groups: Dict[str, FlowGroup], edges: List[FlowEdge]) -> None:
        """Merge inline connectors into the per-flow map."""
        for idx, edge in enumerate(edges):
            flow_name = edge.flow_name or f"edge_{edge.source.component}.{edge.source.port}->{edge.target.component}.{edge.target.port}"
            group = flow_groups.setdefault(flow_name, FlowGroup(flow_name))
            group.add_source(edge.source)
            group.add_target(edge.target)
            group.add_edge(edge.with_flow(flow_name))


# ---------------------------------------------------------------------------
# Simple CLI / demo
# ---------------------------------------------------------------------------


def demo():
    """Run a quick demo on an embedded snippet."""
    sample_mp = """
    /* Sample Ab Initio graph */
    component read_input: Input File {
        out0: FLOW_READ
    }

    component filter_data: Filter by Expression {
        in0: FLOW_READ
        out0: FLOW_FILTERED
    }

    component write_output: Output File {
        in0: FLOW_FILTERED
    }

    flow audit_stream: filter_data.reject -> write_rejects.in0
    flow from read_input to audit_stage
    audit_stage.out0 -> write_output.in1
    """
    parser = AbInitioFlowParser()
    result = parser.parse(sample_mp, source="demo")
    return result


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Extract flows from an Ab Initio .mp file.")
    parser.add_argument("mp_path", nargs="?", help="Path to the .mp file to parse. If omitted, runs demo.")
    args = parser.parse_args()

    flow_parser = AbInitioFlowParser()

    if args.mp_path:
        parsed = flow_parser.parse_file(args.mp_path)
    else:
        parsed = demo()

    json.dump(parsed, sys.stdout, indent=2)
    sys.stdout.write("\n")
