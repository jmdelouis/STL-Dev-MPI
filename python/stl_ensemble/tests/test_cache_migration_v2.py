"""Synthetic cache-integrity/migration checks, without a mock scattering engine.

The feature tables are synthetic fixtures, not claimed to be a native sky
calibration. Native source verification has its own test against the real v1 tree.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from stl_port.config import Config, canonical_hash, sha256_file, BASELINE_REVISION, SOURCE_SHA256
from stl_port.calibration import science_config, population_targets, load_calibration
from stl_port.sc_normalization import SCNormalization
from stl_port.lut import probability_grid
from stl_port import migrate_cache as migration

print('SCRIPT VERSION: test_cache_migration_v2.py 2026-10-29-v2-variance-ratio',flush=True)
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def v1_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('v1_synthetic_calibration')
    config=Config(j=8,bootstrap_seed=20260915,initialization_seed=71022026)
    plan=[dict(family=f,native_dg=0,c1=0,c2=0,l1=o,l3=o)
          for f in ('mean','var','S1','S2','S3','S4')
          for o in (range(4) if f.startswith('S') else [0])]
    rng=np.random.default_rng(101)
    ids=np.random.default_rng(config.bootstrap_seed).choice(np.arange(54),10,replace=True)
    targets={'bootstrap_ids':ids};features={};norms={}
    for label,prefix,scale in (('gaussian','gsc',1.),('physical','psc',1e-5)):
        raw=scale*(rng.normal(size=(54,len(plan)))+1j*rng.normal(size=(54,len(plan))))
        raw[:,0]=scale*rng.normal(0,.01,54)
        raw[:,1]=scale**2*rng.uniform(.5,1.5,54)
        spatial=np.abs(raw)**2+.1*scale**2
        norm=SCNormalization.fit(plan,raw,spatial,nside=config.nside)
        values=norm.forward(raw);norms[label]=norm.spec
        features[prefix]=values;features[prefix+'_raw']=raw;features[prefix+'_spatial_variance']=spatial
        targets[prefix+'_mean'],targets[prefix+'_var']=population_targets(values,ids)
        targets[prefix+'_sigma']=norm.sigma(raw,spatial).mean(0)
    for name,shape in (('power',(54,16)),('pdf',(54,100)),('mf',(54,3,65))):
        v=rng.uniform(.2,2,shape);features[name]=v
        targets[name+'_mean'],targets[name+'_var']=population_targets(v,ids)
    features['gaussian_cl']=rng.uniform(.1,1,(54,384))
    targets['initialization_cl']=features['gaussian_cl'][ids].mean(0)
    targets['initialization_reference_mean']=np.asarray(0.)
    targets['pdf_support']=np.array([-1.,1.]);targets['power_centers']=np.geomspace(2,383,16)
    for name in ('training_physical.npy','training_gaussian.npy'):
        m=np.lib.format.open_memmap(root/name,mode='w+',shape=(54,config.npix),dtype=np.float32)
        m[:]=0;m.flush();del m
    np.savez(root/'targets.npz',**targets);np.savez(root/'training_features.npz',**features)
    p=probability_grid(8192,54*config.npix)
    np.savez(root/'lut.npz',probabilities=p,physical_knots=np.linspace(-.1,.1,len(p),dtype=np.float32))
    library=root/'fixture-native-binary-for-hash-only.so';library.write_bytes(b'not loaded: migration hash fixture')
    known=json.loads((ROOT/'provenance/v1_compatibility.json').read_text())
    oldcfg=science_config(config);oldcfg.pop('sc_variance_contract');oldcfg['script_version']='2026-10-28-v1'
    meta={'library':'STL-Dev-MPI','native_revision':BASELINE_REVISION,'source_sha256':SOURCE_SHA256,
          'config':oldcfg,'plan':plan,'bootstrap_ids':ids.tolist(),'losses':known['losses'],
          'native_library_sha256':sha256_file(library),'implementation_sha256':known['implementation_sha256'],
          'normalizers':norms,'target_sha256':sha256_file(root/'targets.npz'),'lut_sha256':sha256_file(root/'lut.npz'),
          'sources':[{'id':i,'name':f'allskymap_nres12r{i:03d}.zs1.mag.nside2048.fits','sha256':'fixture','bytes':0} for i in range(54)],
          'scientific_variance_guard':{'passed':False,'issues':['synthetic legacy floor guard']}}
    meta['cache_identity']=canonical_hash(meta);meta['variance_floor_report']={}
    (root/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return root,config,library,meta


def clone_small(source,destination):
    destination.mkdir()
    for f in source.iterdir():
        # Sparse NPY banks can be shared within the test; they are never mutated.
        if f.name.startswith('training_') and f.suffix=='.npy':os.link(f,destination/f.name)
        else:shutil.copy2(f,destination/f.name)


def test_migration_preserves_arrays_and_v1_manifest(v1_fixture,tmp_path,monkeypatch):
    source,cfg,lib,old=v1_fixture
    before={name:sha256_file(source/name) for name in migration.ARRAY_FILES+('manifest.json',)}
    monkeypatch.setattr(migration,'verify_v1_package',lambda _: {'fixture_only':True})
    out=tmp_path/'v2'
    new=migration.migrate_v1(cfg,source,out,native_library=lib,v1_package='unused-fixture')
    assert new['cache_identity']!=old['cache_identity']
    for name in migration.ARRAY_FILES:
        assert sha256_file(out/name)==before[name]
        assert sha256_file(source/name)==before[name]
    assert sha256_file(source/'manifest.json')==before['manifest.json']
    assert sha256_file(out/'manifest.v1.json')==before['manifest.json']
    assert new['scientific_variance_guard']['passed']
    assert new['losses']['sc_population_variance']['ratio_floor']==1e-12
    assert new['mean_sigma_report']['method'].startswith('unchanged v1')
    load_calibration(cfg,new['plan'],out,native_library_sha256=sha256_file(lib),require_variance_sensitivity=True)
    with pytest.raises(FileExistsError):
        migration.migrate_v1(cfg,source,out,native_library=lib,v1_package='unused-fixture')
    # The new identity authenticates the reports too, rather than excluding them.
    tampered=copy.deepcopy(new);tampered['mean_sigma_report']['final_absolute_sigma_floor']=1e-30
    (out/'manifest.json').write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='identity'):
        load_calibration(cfg,new['plan'],out,native_library_sha256=sha256_file(lib))


@pytest.mark.parametrize('kind',['target_bytes','manifest_identity','wrong_implementation','wrong_library','wrong_seed'])
def test_refuse_unvalidated_migrations(v1_fixture,tmp_path,monkeypatch,kind):
    source,cfg,lib,meta=v1_fixture
    copyroot=tmp_path/'source';clone_small(source,copyroot)
    monkeypatch.setattr(migration,'verify_v1_package',lambda _: {'fixture_only':True})
    if kind=='target_bytes':
        with (copyroot/'targets.npz').open('ab') as f:f.write(b'tamper')
    elif kind in ('manifest_identity','wrong_implementation'):
        m=copy.deepcopy(meta);m['implementation_sha256']='wrong'
        if kind=='wrong_implementation':
            m['cache_identity']=canonical_hash({k:v for k,v in m.items() if k not in ('cache_identity','variance_floor_report')})
        (copyroot/'manifest.json').write_text(json.dumps(m))
    elif kind=='wrong_library':
        lib=tmp_path/'other.so';lib.write_bytes(b'different binary')
    else:
        import dataclasses
        cfg=dataclasses.replace(cfg,bootstrap_seed=999)
    with pytest.raises((ValueError,RuntimeError)):
        migration.migrate_v1(cfg,copyroot,tmp_path/'out',native_library=lib,v1_package='unused-fixture')
    assert not (tmp_path/'out').exists()


def test_stored_features_checked_against_target_bytes(v1_fixture,tmp_path,monkeypatch):
    source,cfg,lib,meta=v1_fixture
    copyroot=tmp_path/'source';clone_small(source,copyroot)
    with np.load(copyroot/'training_features.npz') as f:features={k:f[k] for k in f.files}
    features['psc']*=2
    np.savez(copyroot/'training_features.npz',**features)
    monkeypatch.setattr(migration,'verify_v1_package',lambda _: {'fixture_only':True})
    with pytest.raises(ValueError,match='consistency'):
        migration.migrate_v1(cfg,copyroot,tmp_path/'out',native_library=lib,v1_package='unused-fixture')


def test_verify_actual_v1_package():
    path=os.environ.get('STL_V1_PACKAGE_DIR')
    if not path:pytest.skip('Actual v1 source tree not specified for provenance test')
    assert migration.verify_v1_package(path)['files_verified']==111


def test_v1_cache_cannot_be_loaded_as_v2(v1_fixture):
    source,cfg,lib,meta=v1_fixture
    with pytest.raises(ValueError,match='explicit v1 cache migration'):
        load_calibration(cfg,meta['plan'],source,native_library_sha256=sha256_file(lib))

@pytest.mark.parametrize('bad',[0.,-1e-25,np.nan])
def test_migration_never_masks_invalid_sc_target(v1_fixture,tmp_path,monkeypatch,bad):
    source,cfg,lib,meta=v1_fixture
    copyroot=tmp_path/'source';clone_small(source,copyroot)
    with np.load(copyroot/'targets.npz') as f:t={k:f[k] for k in f.files}
    t['psc_var'][6]=bad
    np.savez(copyroot/'targets.npz',**t)
    m=copy.deepcopy(meta);m['target_sha256']=sha256_file(copyroot/'targets.npz')
    m['cache_identity']=canonical_hash({k:v for k,v in m.items() if k not in ('cache_identity','variance_floor_report')})
    (copyroot/'manifest.json').write_text(json.dumps(m))
    monkeypatch.setattr(migration,'verify_v1_package',lambda _: {'fixture_only':True})
    with pytest.raises(ValueError,match='positive finite'):
        migration.migrate_v1(cfg,copyroot,tmp_path/'out',native_library=lib,v1_package='unused-fixture')
    assert not (tmp_path/'out').exists()
