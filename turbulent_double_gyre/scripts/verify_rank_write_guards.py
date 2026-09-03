"""Assert every filesystem write in ``train.run`` belongs to rank 0.

Under data parallelism the output tree has exactly one owner. Non-primary ranks
contribute gradients and nothing else: they never create, write or read it. That
is easy to state and easy to violate by adding one line, and the failure does
not appear until a rank reaches the write --- which, after the hour-long
normalizer scans, is an hour into a multi-day job.

So it is checked statically instead. Every write call in ``run`` must either sit
inside an ``if topology.is_primary:`` block, or come after the point where the
non-primary ranks have already returned.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY / "src" / "turbfno" / "train.py"

#: Calls that touch the filesystem, by the attribute name they are called on.
WRITES = {
    "savez", "savez_compressed", "save", "write_text", "write_bytes",
    "mkdir", "copy2", "copy", "copyfile", "rmtree", "replace", "unlink",
    "touch", "rename",
}


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_primary_test(test: ast.expr) -> bool:
    """A test that holds only on rank 0.

    Either ``topology.is_primary`` itself, or an ``and`` chain containing it --
    ``topology.is_primary and not resume_validation`` narrows the guard further,
    so it is still sound. An ``or`` chain is not: one of its branches can be
    true on a non-primary rank.
    """

    if (
        isinstance(test, ast.Attribute)
        and test.attr == "is_primary"
        and isinstance(test.value, ast.Name)
        and test.value.id == "topology"
    ):
        return True
    return (
        isinstance(test, ast.BoolOp)
        and isinstance(test.op, ast.And)
        and any(_is_primary_test(value) for value in test.values)
    )


def _is_not_primary_test(test: ast.expr) -> bool:
    """``not topology.is_primary`` -- the early-return / skip form."""

    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _is_primary_test(test.operand)
    )


def _scope_of(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    """Nearest enclosing function or loop -- the region a handover covers."""

    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.For, ast.While, ast.AsyncFor)):
            return current
        current = parents.get(id(current))
    return None


def main() -> int:
    tree = ast.parse(SOURCE.read_text())
    run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(run):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    # Lines inside an `if topology.is_primary:` body.
    guarded: set[int] = set()
    # Handovers: `if not topology.is_primary:` bodies that leave -- by return,
    # by raise, or by continue.  Past one, only the primary rank still runs the
    # rest of its enclosing scope.  Recorded as (line, enclosing scope).
    handovers: list[tuple[int, ast.AST | None]] = []

    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue
        if _is_primary_test(node.test):
            for statement in node.body:
                for inner in ast.walk(statement):
                    if hasattr(inner, "lineno"):
                        guarded.add(inner.lineno)
        elif _is_not_primary_test(node.test):
            leaves = any(
                isinstance(inner, (ast.Return, ast.Raise, ast.Continue))
                for statement in node.body
                for inner in ast.walk(statement)
            )
            if leaves:
                last = max(
                    getattr(inner, "lineno", 0)
                    for statement in node.body
                    for inner in ast.walk(statement)
                )
                handovers.append((last, _scope_of(node, parents)))

    unguarded: list[tuple[int, str]] = []
    for node in ast.walk(run):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in WRITES:
            continue
        if node.lineno in guarded:
            continue
        scope = _scope_of(node, parents)
        covered = False
        for line, handover_scope in handovers:
            if node.lineno <= line:
                continue
            # The handover covers this write if the write sits in the same
            # scope, or in one nested inside it.
            walker: ast.AST | None = scope
            while walker is not None:
                if walker is handover_scope:
                    covered = True
                    break
                walker = parents.get(id(walker))
            if handover_scope is None or covered:
                covered = True
                break
        if not covered:
            unguarded.append((node.lineno, name))

    print(f"{SOURCE.relative_to(STUDY)}: {len(handovers)} rank handover(s) at "
          f"lines {sorted(line for line, _ in handovers)}")
    if unguarded:
        for line, name in sorted(unguarded):
            print(f"  UNGUARDED WRITE  line {line}: {name}(...)")
        print("RANK WRITE GUARDS: FAIL")
        return 1
    print("  every write is guarded or past a handover")
    print("RANK WRITE GUARDS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
