from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "parselabs"


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
    return ".".join(relative.parts)


def _collect_import_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}

    for path in PACKAGE_DIR.rglob("*.py"):
        module_name = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        edges: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("parselabs."):
                        edges.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "parselabs":
                    for alias in node.names:
                        if alias.name != "*":
                            edges.add(f"parselabs.{alias.name}")
                elif node.module and node.module.startswith("parselabs"):
                    edges.add(node.module)

        graph[module_name] = edges

    return graph


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visited: set[str] = set()
    active: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in active:
            cycle_start = stack.index(node)
            return stack[cycle_start:] + [node]
        if node in visited:
            return None

        visited.add(node)
        active.add(node)
        stack.append(node)

        for neighbor in sorted(graph.get(node, set())):
            if neighbor not in graph:
                continue
            cycle = visit(neighbor)
            if cycle is not None:
                return cycle

        stack.pop()
        active.remove(node)
        return None

    for node in sorted(graph):
        cycle = visit(node)
        if cycle is not None:
            return cycle

    return None


def test_review_import_graph_has_no_cycles():
    graph = _collect_import_graph()
    cycle = _find_cycle(graph)
    assert cycle is None, f"unexpected package import cycle: {' -> '.join(cycle or [])}"


def test_review_state_does_not_depend_on_review_ui_layers():
    graph = _collect_import_graph()
    assert "parselabs.review" not in graph["parselabs.review_state"]
    assert "parselabs.results_view" not in graph["parselabs.review_state"]
    assert "parselabs.document_reviewer" not in graph["parselabs.review_state"]


def test_standardization_refresh_does_not_depend_on_extraction_layer():
    graph = _collect_import_graph()
    assert "parselabs.extraction" not in graph["parselabs.standardization_refresh"]
