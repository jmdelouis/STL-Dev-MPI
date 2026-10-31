"""Regression tests at the measured physical scales, not just order-one features."""
import json
from pathlib import Path
import numpy as np
import pytest
from stl_port.losses import moments, moments_vjp, sc_loss, equalize_sigma
from stl_port.sc_variance import variance_ratio_loss, RATIO_FLOOR
from stl_port.sc_reports import variance_sensitivity_issues, mean_sigma_report

print("SCRIPT VERSION: test_variance_ratio_v2.py 2026-10-29-v2-variance-ratio", flush=True)


def plan():
    return [dict(family=f, l1=o, l3=o, native_dg=0, c1=0, c2=0)
            for f in ('mean','var','S1','S2','S3','S4')
            for o in (range(4) if f.startswith('S') else [0])]


@pytest.mark.parametrize('reference', [1e-28, 3.96803234e-28, 2.64500443e-25, 1.76484203e-23,
                                       2.50117496e-21, 1e-12, 1.01735745e-11, 1e-4, 1.])
@pytest.mark.parametrize('ratio', [0.1, 1., 10.])
def test_known_ratios_at_all_amplitudes(reference, ratio):
    v=np.array([reference*ratio]);t=np.array([reference])
    loss,cot=variance_ratio_loss(v,t)
    np.testing.assert_allclose(loss,np.log(ratio)**2,rtol=2e-13,atol=3e-28)
    np.testing.assert_allclose(cot*v,2*np.log(ratio),rtol=2e-13,atol=5e-14)


@pytest.mark.parametrize('complex_values', [False, True])
def test_physical_scale_variance_only_finite_differences(complex_values):
    rng=np.random.default_rng(591)
    scale=np.logspace(-14,0,24)
    x=rng.normal(size=(10,24))
    d=rng.normal(size=x.shape)
    if complex_values:
        x=x+1j*rng.normal(size=x.shape);d=d+1j*rng.normal(size=x.shape)
    x*=scale;d*=scale
    _,v=moments(x);t=v*np.geomspace(.13,7.,24)
    def fun(y):
        mu,var=moments(y);loss,dv=variance_ratio_loss(var,t)
        return loss,moments_vjp(y,np.zeros_like(mu),dv)
    loss,g=fun(x);analytic=np.vdot(g,d).real
    errors=[]
    for h in (1e-4,1e-5,1e-6):
        fd=(fun(x+h*d)[0]-fun(x-h*d)[0])/(2*h)
        errors.append(abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12))
    assert min(errors)<2e-6, errors
    assert np.isfinite(g).all() and np.linalg.norm(g)>0


@pytest.mark.parametrize('complex_values', [False, True])
def test_invariance_under_per_coefficient_units(complex_values):
    rng=np.random.default_rng(3);x=rng.normal(size=(10,19))
    if complex_values:x=x+1j*rng.normal(size=x.shape)
    _,v=moments(x);t=v*np.linspace(.1,10,19)
    def f(y,ref):
        mu,var=moments(y);l,dv=variance_ratio_loss(var,ref)
        return l,moments_vjp(y,np.zeros_like(mu),dv)
    l,g=f(x,t)
    a=np.logspace(-14,6,19)
    ll,gg=f(x*a,t*a*a)
    np.testing.assert_allclose(ll,l,rtol=1e-13)
    np.testing.assert_allclose(gg*a,g,rtol=2e-12,atol=1e-14)


@pytest.mark.parametrize('bad', [0., -1e-28, np.nan, np.inf])
def test_invalid_selected_targets_fail(bad):
    with pytest.raises(ValueError):variance_ratio_loss([1e-25],[bad])


@pytest.mark.parametrize('bad', [-1e-25, np.nan, np.inf])
def test_invalid_synthesis_variance_fails(bad):
    with pytest.raises(ValueError):variance_ratio_loss([bad],[1e-25])


def test_hard_relative_floor_and_zero_synthesis():
    t=np.array([1e-28,1e-12,1.])
    l,g=variance_ratio_loss(np.zeros(3),t)
    np.testing.assert_allclose(l,np.log(RATIO_FLOOR)**2)
    np.testing.assert_array_equal(g,0)
    l,g=variance_ratio_loss(t*(RATIO_FLOOR/2),t)
    np.testing.assert_array_equal(g,0)
    l,g=variance_ratio_loss(t*(RATIO_FLOOR*2),t)
    assert np.all(g!=0)
    # Ratio would overflow; the log-difference remains defined.
    l,g=variance_ratio_loss([1e200],[1e-200])
    np.testing.assert_allclose(l,(400*np.log(10))**2,rtol=1e-14)
    assert np.isfinite(g).all()


def test_no_absolute_dead_zone_and_plan_permutation():
    p=plan();rng=np.random.default_rng(81);x=1e-14*rng.normal(size=(10,len(p)))
    mu,v=moments(x);t=v*10;s=np.ones(len(p))
    loss,g=sc_loss(x,mu,t,s,p,physical=True)
    np.testing.assert_allclose(loss,np.log(10)**2,rtol=1e-13)
    assert np.linalg.norm(g)>0
    perm=rng.permutation(len(p))
    ll,gg=sc_loss(x[:,perm],mu[perm],t[perm],s[perm],[p[i] for i in perm],physical=True)
    np.testing.assert_allclose(ll,loss,rtol=1e-14)
    np.testing.assert_allclose(gg,g[:,perm],rtol=2e-13)


@pytest.mark.parametrize('physical', [False,True])
def test_mean_loss_and_sigma_unchanged_from_v1(physical):
    # Execute the archived v1 functions verbatim, solely for an independent reference.
    ns={};src=Path(__file__).resolve().parents[1]/'provenance/v1_losses.py.txt'
    exec(compile(src.read_text(),str(src),'exec'),ns)
    p=plan();rng=np.random.default_rng(31);x=1e-13*(rng.normal(size=(10,len(p)))+1j*rng.normal(size=(10,len(p))))
    mu,v=moments(x);target=mu+np.linspace(1e-14,1e-13,len(p))
    sigma=np.geomspace(1e-15,1e-10,len(p))
    # v1's variance term is exactly zero and has zero cotangent at these scales.
    old_loss,old_cot=ns['sc_loss'](x,target,v,sigma,p,physical=physical)
    new_loss,new_cot=sc_loss(x,target,v,sigma,p,physical=physical)
    np.testing.assert_array_equal(equalize_sigma(sigma,p),ns['equalize_sigma'](sigma,p))
    np.testing.assert_allclose(new_loss,old_loss,rtol=1e-15,atol=0)
    np.testing.assert_array_equal(new_cot,old_cot)


def test_all_gaussian_entries_checked_physical_s0_not_registered():
    p=plan();t={'gsc_var':np.ones(len(p))*1e-28,'psc_var':np.ones(len(p))*1e-28}
    assert variance_sensitivity_issues(t,p)==[]
    t['psc_var'][0]=0
    assert variance_sensitivity_issues(t,p)==[]
    t['gsc_var'][0]=0
    assert any('gsc/mean' in s for s in variance_sensitivity_issues(t,p))


def test_mean_sigma_report_is_read_only_and_json_safe():
    p=plan();sigma=np.geomspace(1e-15,1e-6,len(p));t={'gsc_sigma':sigma.copy(),'psc_sigma':sigma.copy()}
    r=mean_sigma_report(t,p)
    assert r['gsc']['S1']['strictly_below_final_floor_before_clamp']>0
    json.dumps(r,allow_nan=False)
    np.testing.assert_array_equal(t['psc_sigma'],sigma)
