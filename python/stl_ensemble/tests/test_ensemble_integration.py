"""Real-native integration test; skips explicitly when the bridge is unavailable."""
import importlib.util
import os
from pathlib import Path
import sys
import pytest

print("SCRIPT VERSION: test_ensemble_integration.py 2026-10-28-v1",flush=True)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def native_available():
    choices=[os.environ.get("STL_NATIVE_LIBRARY",""),str(ROOT/"native/libstl_bridge.so"),str(ROOT.parent/"STL-Dev-MPI/target/release/libstl_bridge.so")]
    return any(p and Path(p).is_file() for p in choices)

@pytest.mark.skipif(not native_available(),reason="Real native STL shared library has not been built; no mock scattering substitute")
def test_full_five_block_execution_equivalence(tmp_path):
    from stl_port.config import Config
    from validate_modes import run_validation
    config=Config(nside=8,backend="cpu",threads=1,ps_variance_upper=23,cache_bytes=256*1024**2)
    report=run_validation(config,tmp_path)
    assert report["passed"], report
    for mode,step in report["optimizer_step_validation"].items():
        assert step["nonzero_step"], (mode,step)
        assert step["loss_decreased"], (mode,step)

def test_grf_matches_unchanged_source_rng_and_mean_convention():
    import ast
    import numpy as np
    import healpy as hp
    from stl_port.calibration import draw_initial_ensemble
    from stl_port.config import Config
    reference=ROOT/"reference/05-Multireso_v41p23z1_STEP1ONLY_mean_gradient_v2.py"
    if not reference.is_file(): reference=ROOT.parents[1]/"project_sources/05-Multireso_v41p23z1_STEP1ONLY_mean_gradient_v2.py"
    if not reference.is_file(): pytest.skip("Unchanged source script not present for direct reference initializer test")
    tree=ast.parse(reference.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_draw_target_cl_grf_batch_nest")
    env={"np":np,"hp":hp}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(reference),"exec"),env)
    cfg=Config(nside=8,backend="cpu",threads=1,ps_variance_upper=23)
    rng=np.random.default_rng(42)
    target=rng.normal(size=cfg.npix)+.17
    cl=np.linspace(.12,.001,cfg.lmax+1)
    expected,_=env[node.name](target,cfg.nside,10,cfg.initialization_seed,exact_cl=False,match_mean_std=False,remove_monopole=True,cl_floor_rel=0,cl_target_override=cl)
    actual=draw_initial_ensemble(cfg,{"initialization_cl":cl,"initialization_reference_mean":target.mean()})
    np.testing.assert_allclose(actual,expected.astype(np.float64),rtol=8e-7,atol=2e-7)
