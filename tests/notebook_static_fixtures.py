"""AST extraction and helper-module unit fixtures, never notebook execution.

These tests may call CPU helper functions. They cannot certify a notebook,
its cell order, installation, rendering or GPU execution; acceptance is in Colab.
"""
import ast
import json


def code_cells(notebook):
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def recorded_literal(notebook, name):
    matches = []
    for source in code_cells(notebook):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                call = node.value
                assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                assert isinstance(call.func.value, ast.Name) and call.func.value.id == 'json'
                assert call.func.attr == 'loads' and isinstance(call.args[0], ast.Constant)
                matches.append(call.args[0].value)
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def recorded_json(notebook, name):
    return json.loads(recorded_literal(notebook, name))


def helper_module(notebook, namespace_name, dependencies=None):
    """Import exactly one literal helper module; do not execute its notebook cell."""
    matches = []
    for source in code_cells(notebook):
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'exec' and len(node.args) == 2
                and isinstance(node.args[1], ast.Name) and node.args[1].id == namespace_name):
                assert isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                matches.append(node.args[0].value)
    assert len(matches) == 1, (namespace_name, len(matches))
    namespace = {'__name__': 'unit_fixture_' + namespace_name, **(dependencies or {})}
    exec(compile(matches[0], 'embedded-helper-unit-fixture', 'exec'), namespace)
    return namespace
