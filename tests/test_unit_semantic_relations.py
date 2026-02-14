from __future__ import annotations

from accel.query.semantic_relations import (
    build_semantic_relation_graph,
    collect_seed_files,
    compute_relation_proximity,
    expand_candidates_with_relations,
)


def test_build_semantic_relation_graph_resolves_reference_targets() -> None:
    symbols_by_file = {
        "src/a.py": [{"symbol": "Alpha", "qualified_name": "module.Alpha"}],
        "src/b.py": [{"symbol": "Beta", "qualified_name": "module.Beta"}],
    }
    references_by_file = {
        "src/caller.py": [{"target_symbol": "Alpha"}],
    }
    deps_by_file = {
        "src/caller.py": [{"edge_to": "src.a"}],
    }
    graph = build_semantic_relation_graph(
        symbols_by_file=symbols_by_file,
        references_by_file=references_by_file,
        deps_by_file=deps_by_file,
        indexed_files=["src/a.py", "src/b.py", "src/caller.py"],
    )

    assert "src/caller.py" in graph
    assert float(graph["src/caller.py"].get("src/a.py", 0.0)) > 0.0


def test_expand_candidates_with_relations_prioritizes_related_nodes() -> None:
    candidates = ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]
    graph = {
        "src/a.py": {"src/d.py": 1.0},
        "src/d.py": {"src/a.py": 1.0},
    }
    seeds = collect_seed_files(
        changed_files=["src/a.py"],
        hints=[],
        candidate_files=candidates,
    )
    expanded = expand_candidates_with_relations(
        candidates=candidates,
        relation_graph=graph,
        seed_files=seeds,
        changed_files=["src/a.py"],
    )

    assert expanded[0] == "src/a.py"
    assert expanded[1] == "src/d.py"


def test_compute_relation_proximity_uses_reverse_link() -> None:
    graph = {
        "src/seed.py": {"src/target.py": 0.7},
    }
    score = compute_relation_proximity(
        file_path="src/target.py",
        relation_graph=graph,
        seed_files=["src/seed.py"],
    )
    assert score > 0.0
