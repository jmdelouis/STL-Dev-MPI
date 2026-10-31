"""Exact joint ten-member five-block objective, with one native device workspace."""
from __future__ import annotations
import time
import numpy as np
from .losses import sc_loss, power_loss, pdf_loss, mf_loss
from .harmonics import FullSkySHT, lp_windows
from .minkowski import Minkowski
from .pdf import CICPDF

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: ensemble.py {SCRIPT_VERSION}")

class EnsembleObjective:
    """A scalar/gradient pair for all B maps; the optimizer never sees partial updates.

    SC normalizers expose forward(raw), vjp(raw,cot) and sigma(raw,spatial_var).
    Native SC checkpoints exist only during their own objective's pass. MF state
    is replayed one member at a time; power alms are retained on the CPU and
    tagged with generation/member/objective by FullSkySHT.
    """
    def __init__(self, config, engine, targets, lut, gaussian_normalizer,
                 physical_normalizer, *, scientific=True):
        config.validate(scientific=scientific)
        if scientific:
            from .calibration import variance_sensitivity_issues
            issues=variance_sensitivity_issues(targets,engine.plan)
            if issues: raise ValueError("Invalid SC variance-ratio targets; all selected TRAIN54 variances must be finite and positive: "+"; ".join(issues))
        self.config, self.engine, self.targets, self.lut = config, engine, targets, lut
        self.normalizers={"gaussian":gaussian_normalizer,"physical":physical_normalizer}
        self.sht=FullSkySHT(config.nside,lmax=config.lmax,nthreads=config.threads)
        self.windows,self.centers=lp_windows(config.lmax)
        self.pdf=CICPDF(float(targets["pdf_support"][0]),float(targets["pdf_support"][1]))
        self.mf=Minkowski(self.sht)
        self.generation=0
        self.last=None
        self.evaluation_callback=None
        if engine.batch != config.active_batch:
            raise ValueError("Native engine must be sized correctly at construction, before evaluating any map")

    def _physical(self, g):
        return self.lut.inverse(g,remove_monopole=True)

    def _sc_block(self, maps, objective):
        cfg=self.config; gen=self.generation; B=cfg.members
        physical=objective=="physical"
        norm=self.normalizers[objective]
        prefix="psc" if physical else "gsc"
        t=self.targets
        bank=[]; raw=[]
        checkpoint_capture_s=0.; checkpoint_restore_s=0.; replay_s=0.; native_reverse_s=0.
        started=time.perf_counter()
        if cfg.mode=="resident":
            input_maps=self._physical(maps)[0] if physical else maps
            raw_values,_=self.engine.forward(input_maps,generation=gen,member=cfg.members,objective=objective)
        else:
            for b in range(B):
                input_map=self._physical(maps[b])[0] if physical else maps[b]
                value,_=self.engine.forward(input_map[None,:],generation=gen,member=b,objective=objective)
                raw.append(value[0].copy())
                if cfg.mode=="stream-checkpoint":
                    tick=time.perf_counter(); bank.append(self.engine.checkpoint())
                    checkpoint_capture_s+=time.perf_counter()-tick
            raw_values=np.stack(raw)
        forward_seconds=time.perf_counter()-started
        values=norm.forward(raw_values)
        details={}
        scalar,cot=sc_loss(values,t[prefix+"_mean"],t[prefix+"_var"],t[prefix+"_sigma"],self.engine.plan,physical=physical,diagnostics=details)
        self._sc_diagnostics[objective]=details
        raw_cot=norm.vjp(raw_values,cot)
        gradient=np.empty_like(maps,dtype=np.float64)
        started=time.perf_counter()
        if cfg.mode=="resident":
            tick=time.perf_counter()
            gradient[:]=self.engine.vjp(raw_cot,generation=gen,member=cfg.members,objective=objective)
            native_reverse_s+=time.perf_counter()-tick
            if physical:
                for b in range(B):
                    _,state=self._physical(maps[b])
                    gradient[b]=self.lut.vjp(state,gradient[b])
        else:
            for b in range(B):
                if cfg.mode=="stream-checkpoint":
                    tick=time.perf_counter()
                    self.engine.restore(bank[b],generation=gen,member=b,objective=objective)
                    checkpoint_restore_s+=time.perf_counter()-tick
                    bank[b].close()
                    bank[b]=None  # Release this member's CPU state as soon as it is consumed.
                else:
                    input_map=self._physical(maps[b])[0] if physical else maps[b]
                    tick=time.perf_counter()
                    replay,_=self.engine.forward(input_map[None,:],generation=gen,member=b,objective=objective)
                    replay_s+=time.perf_counter()-tick
                    if not np.array_equal(replay[0],raw_values[b]):
                        # Deterministic native reductions must replay identically for identical data.
                        if not np.allclose(replay[0],raw_values[b],rtol=1e-12,atol=1e-14):
                            raise RuntimeError("Native replay statistics differ from the current trial")
                tick=time.perf_counter()
                gradient[b]=self.engine.vjp(raw_cot[b:b+1],generation=gen,member=b,objective=objective)[0]
                native_reverse_s+=time.perf_counter()-tick
                if physical:
                    _,state=self._physical(maps[b])
                    gradient[b]=self.lut.vjp(state,gradient[b])
        bank.clear()
        return scalar,gradient,values,cot,{"statistics_and_checkpoint_s":forward_seconds,
            "statistics_forward_excluding_checkpoint_s":forward_seconds-checkpoint_capture_s,
            "reverse_and_replay_s":time.perf_counter()-started,"checkpoint_capture_s":checkpoint_capture_s,
            "checkpoint_restore_s":checkpoint_restore_s,"native_forward_replay_s":replay_s,"native_reverse_s":native_reverse_s}

    def __call__(self, x):
        cfg=self.config
        maps=np.asarray(x,dtype=np.float64).reshape(cfg.members,cfg.npix)
        if not np.all(np.isfinite(maps)): raise ValueError("Trial maps contain non-finite values")
        self.generation+=1
        gen=self.generation
        start=time.perf_counter()
        native_before=self.engine.info().get("timers",{})
        blocks={}; statistics={}; cotangents={}; timing={}
        self._sc_diagnostics={}
        loss,gradient,values,cot,phase=self._sc_block(maps,"gaussian")
        blocks["gaussian_sc"]=loss; statistics["gaussian_sc"]=values; cotangents["gaussian_sc"]=cot; timing["gaussian_sc"]=phase
        loss,g,values,cot,phase=self._sc_block(maps,"physical")
        gradient+=g
        del g
        blocks["physical_sc"]=loss; statistics["physical_sc"]=values; cotangents["physical_sc"]=cot; timing["physical_sc"]=phase
        # Compact statistics and alms only: no bank of ten MF derivative maps.
        phase_start=time.perf_counter()
        alm_bank=[]; powers=[]; pdfs=[]; mfs=[]; outside=[]
        for phase in ("power_forward_s","power_adjoint_s","pdf_forward_s","pdf_reverse_s","mf_forward_s","mf_replay_s","mf_reverse_s","other_lut_forward_s","other_lut_pullback_s"):
            timing[phase]=0.
        for b in range(cfg.members):
            tick=time.perf_counter(); physical,_=self._physical(maps[b])
            timing["other_lut_forward_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            alm=self.sht.cache_analysis(physical,generation=gen,member=b,objective="physical-power")
            alm_bank.append(alm)
            powers.append(self.windows @ self.sht.power(alm.alm))
            timing["power_forward_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            hist,pdfstate=self.pdf.forward(physical)
            pdfs.append(hist)
            outside.append(getattr(pdfstate,"outside_fraction",None))
            timing["pdf_forward_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            mf,mfstate=self.mf.forward(physical)
            mfs.append(mf)
            timing["mf_forward_s"]+=time.perf_counter()-tick
            del mfstate
        powers,pdfs,mfs=np.stack(powers),np.stack(pdfs),np.stack(mfs)
        for name,data,fn in (("power",powers,power_loss),("pdf",pdfs,pdf_loss),("mf",mfs,mf_loss)):
            args=(self.centers,) if name=="power" else ()
            kw={"cutoff":cfg.ps_variance_upper} if name=="power" else {}
            blocks[name],cotangents[name]=fn(data,self.targets[name+"_mean"],self.targets[name+"_var"],*args,**kw)
            statistics[name]=data
        timing["other_forward_s"]=time.perf_counter()-phase_start
        phase_start=time.perf_counter()
        for b in range(cfg.members):
            tick=time.perf_counter(); physical,lutstate=self._physical(maps[b])
            timing["other_lut_forward_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            gp=self.sht.cached_power_vjp(alm_bank[b],self.windows.T @ cotangents["power"][b],generation=gen,member=b,objective="physical-power")
            alm_bank[b]=None
            timing["power_adjoint_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            _,pdfstate=self.pdf.forward(physical)
            gp+=self.pdf.vjp(pdfstate,cotangents["pdf"][b])
            timing["pdf_reverse_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            _,mfstate=self.mf.forward(physical)
            timing["mf_replay_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            gp+=self.mf.vjp(mfstate,cotangents["mf"][b])
            timing["mf_reverse_s"]+=time.perf_counter()-tick
            del mfstate
            tick=time.perf_counter()
            gradient[b]+=self.lut.vjp(lutstate,gp)
            timing["other_lut_pullback_s"]+=time.perf_counter()-tick
        timing["other_reverse_and_mf_replay_s"]=time.perf_counter()-phase_start
        # Exactly five registered blocks. B has already entered moment cotangents.
        total=sum(blocks.values())/5.0
        gradient/=5.0
        if not np.isfinite(total) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Non-finite five-block objective or gradient")
        native_after=self.engine.info().get("timers",{})
        timing["native_phase_delta_s"]={k:float(v)-float(native_before.get(k,0)) for k,v in native_after.items()}
        self.last={"generation":gen,"loss":float(total),"blocks":blocks,
                   "statistics":statistics,"cotangents":cotangents,
                   "sc_loss_diagnostics":self._sc_diagnostics,
                   "timing":timing,"elapsed_s":time.perf_counter()-start,
                   "gradient_norm":float(np.linalg.norm(gradient)),"pdf_outside_fraction":outside}
        if self.evaluation_callback is not None: self.evaluation_callback(self.last)
        return float(total),gradient.reshape(np.shape(x))
