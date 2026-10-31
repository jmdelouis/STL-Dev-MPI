"""Protect the approved scope: native engine, optimizer and other losses unchanged."""
from pathlib import Path
import ast
import hashlib
import json

print('SCRIPT VERSION: test_scope_v2.py 2026-10-29-v2-variance-ratio',flush=True)
ROOT=Path(__file__).resolve().parents[1]


def test_native_scattering_bridge_optimizer_and_feature_map_byte_identical():
    expected=json.loads((ROOT/'provenance/v1_PACKAGE_SHA256.json').read_text())
    selected=[k for k in expected if k.startswith('STL-Dev-MPI/') or k in (
        'stl_port/native.py','stl_port/sc_normalization.py','stl_port/harmonics.py',
        'stl_port/lut.py','stl_port/pdf.py','stl_port/minkowski.py')]
    assert selected
    for path in selected:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected[path],path


def test_mean_sigma_and_non_sc_loss_functions_unchanged():
    old=ast.parse((ROOT/'provenance/v1_losses.py.txt').read_text())
    new=ast.parse((ROOT/'stl_port/losses.py').read_text())
    def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    for name in ('torch_median','moments','moments_vjp','_targets','equalize_sigma','power_loss','pdf_loss','mf_loss'):
        assert ast.dump(a[name])==ast.dump(b[name]),name
    # SC mean/normalization statements, up to the old variance penalty, unchanged.
    prefix=[]
    for node in a['sc_loss'].body[1:]:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='difference' for t in node.targets):break
        prefix.append(node)
    assert [ast.dump(n) for n in prefix]==[ast.dump(n) for n in b['sc_loss'].body[1:1+len(prefix)]]
