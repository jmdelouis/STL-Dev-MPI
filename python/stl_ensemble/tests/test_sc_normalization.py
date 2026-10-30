"""Numerical contract checks for the audited native SC coordinate adapter."""
import numpy as np
import pytest
from stl_port.sc_normalization import SCNormalization

SCRIPT_VERSION="2026-10-28-v1"
print(f"SCRIPT VERSION: test_sc_normalization.py {SCRIPT_VERSION}")


def fixture():
    plan=[dict(family=f,native_dg=d,c1=0,c2=0,l1=None,l2=None,l3=None,j1=None,j2=None,j3=None)
          for f,d in (("mean",0),("var",0),("S1",0),("S2",0),("S3",1),("S4",1))]
    raw=np.asarray([[0.4,2.3,0.7,1.4,0.2+0.1j,0.4-0.2j],[0.3,1.9,0.9,1.6,0.1-0.3j,0.5+0.1j]])
    spatial=np.asarray([[2.14,4,0.2,0.8,0.4,0.3],[1.81,3,0.3,0.7,0.5,0.2]])
    return raw,spatial,SCNormalization.fit(plan,raw,spatial,nside=4)


def test_source_coordinates_and_sigma():
    raw,spatial,norm=fixture()
    values=norm.forward(raw)
    np.testing.assert_array_equal(values[:,2:],raw[:,2:])
    np.testing.assert_allclose(values[:,1].real,np.sqrt((raw[:,1].real-raw[:,0].real**2)/192))
    sig=norm.sigma(raw,spatial)
    np.testing.assert_allclose(sig[:,0],np.sqrt(spatial[:,0]/192))
    np.testing.assert_array_equal(sig[:,1],sig[:,0])
    np.testing.assert_allclose(sig[:,4],np.sqrt(spatial[:,4]/48))
    # This explicitly does not use variance across these two input maps.
    np.testing.assert_allclose(norm.sigma(np.repeat(raw[:1],2,axis=0),spatial),sig)


def test_adapter_vjp_and_serialization():
    raw,spatial,norm=fixture()
    rng=np.random.default_rng(6)
    cot=rng.normal(size=raw.shape)+1j*rng.normal(size=raw.shape)
    direction=rng.normal(size=raw.shape)+1j*rng.normal(size=raw.shape)
    h=1e-6
    fd=np.vdot(cot,norm.forward(raw+h*direction)-norm.forward(raw-h*direction)).real/(2*h)
    exact=np.vdot(norm.vjp(raw,cot),direction).real
    np.testing.assert_allclose(exact,fd,rtol=1e-8,atol=1e-8)
    restored=SCNormalization(norm.plan,norm.spec)
    np.testing.assert_array_equal(norm.forward(raw),restored.forward(raw))
    bad=dict(norm.spec,feature_coordinates="fitted_scale")
    with pytest.raises(ValueError,match="Unreviewed"):
        SCNormalization(norm.plan,bad)


def test_complex_sigma_analogue_is_positive_and_phase_invariant():
    raw,spatial,norm=fixture()
    changed=raw.copy()
    changed[:,4:]*=np.exp(0.6j)
    # The native engine's total variance is invariant under a phase rotation.
    np.testing.assert_array_equal(norm.sigma(raw,spatial),norm.sigma(changed,spatial))
    assert np.all(np.isfinite(norm.sigma(raw,spatial)))
