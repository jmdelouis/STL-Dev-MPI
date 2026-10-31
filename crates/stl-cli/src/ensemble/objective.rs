//! The five-block ensemble objective of P. Campeti
//! (`stl_port/ensemble.py`, `EnsembleObjective`), with the members' maps as
//! the variable:
//!
//! ```text
//! L = (L_GSC + L_PSC + L_PS + L_PDF + L_MF) / 5
//! ```
//!
//! on the Gaussian maps `g` of the B members and their physical maps
//! `kappa = f(g) - mean(f(g))`: the SC statistics of g and of kappa through
//! the engine (one batch of this group's members, CPU or device), the C_l
//! bands, the CIC PDF and the Minkowski functionals of kappa on the host.
//! Every block compares the mean over the members with the target mean and
//! their variance with the population variance.
//!
//! Members are split over groups of ranks as in `stl-synth` (`--groups`):
//! the features of every member are gathered over `cross` before the
//! losses.  The host terms need whole maps: every rank of a group gathers
//! its group's maps and computes them (no gain from several ranks per group
//! for those terms).

use super::calibration::Calibration;
use super::fields::{band_powers, band_powers_vjp, lp_windows, par_chunks, par_zip, Cic, Minkowski, MinkowskiState};
#[cfg(feature = "gpu")]
use super::gpu_terms::GpuTerms;
use super::losses::{mf_loss, pdf_loss, power_loss, sc_loss, Block, ScCoordinates};
use num_complex::Complex64;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;
use stl_core::plan::Coefficient;
use stl_core::sht::Sht;
use stl_mpi::synthesis::Objective;
use stl_mpi::Comm;

type C64 = Complex64;

/// Blocks of the last evaluation, in the order GSC, PSC, PS, PDF, MF.
pub const BLOCKS: [&str; 5] = ["gaussian_sc", "physical_sc", "power", "pdf", "mf"];

/// Host memory for the Minkowski states of a group's members between the
/// forward and the reverse pass (4 GiB: nside 2048 recomputes them).
const MF_STATE_BYTES: usize = 4 << 30;

pub struct EnsembleObjective<'o, 'a, C: Comm> {
    /// SC of g and of kappa: two objectives on the same partition (and device engine).
    pub g: &'o Objective<'a, C>,
    pub k: &'o Objective<'a, C>,
    pub cross: Option<&'o C>,
    pub cal: &'o Calibration,
    pub plan: &'o [Coefficient],
    pub members: usize,
    pub first_member: usize,
    coords: ScCoordinates,
    sht: Sht,
    windows: Vec<Vec<f64>>,
    centers: Vec<f64>,
    cic: Cic,
    mk: Minkowski,
    /// Upper band centre of the C_l variance term.
    cutoff: f64,
    npix: usize,
    pub last: RefCell<[Block; 5]>,
    pub timers: RefCell<BTreeMap<&'static str, f64>>,
    #[cfg(feature = "gpu")]
    gpu_terms: Option<RefCell<GpuTerms>>,
}

/// What the host's cotangents of the harmonic terms need of the forward pass.
#[derive(Default)]
struct HostPass {
    alms: Vec<Vec<C64>>,
    states: Vec<MinkowskiState>,
}

impl<'o, 'a, C: Comm> EnsembleObjective<'o, 'a, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g: &'o Objective<'a, C>,
        k: &'o Objective<'a, C>,
        cross: Option<&'o C>,
        cal: &'o Calibration,
        plan: &'o [Coefficient],
        members: usize,
        first_member: usize,
        cutoff: f64,
    ) -> Result<Self, String> {
        let nside = cal.nside;
        if plan.len() != cal.plan_len || g.geo.pixels(0) != 12 * nside * nside {
            return Err("calibration of another plan or nside".into());
        }
        if members < 2 {
            return Err("the population losses need at least two members".into());
        }
        let lmax = 3 * nside - 1;
        let (windows, centers) = lp_windows(lmax);
        if !(20.0..=lmax as f64).contains(&cutoff) {
            return Err(format!("C_l variance cutoff {cutoff} outside [20, {lmax}]"));
        }
        let t = &cal.targets;
        Ok(EnsembleObjective {
            g,
            k,
            cross,
            cal,
            plan,
            members,
            first_member,
            coords: ScCoordinates::new(plan, nside)?,
            sht: Sht::new(nside, lmax),
            windows,
            centers,
            cic: Cic::new(t.pdf_support[0], t.pdf_support[1], 100),
            mk: Minkowski::default(),
            cutoff,
            npix: 12 * nside * nside,
            last: RefCell::new([Block::default(); 5]),
            timers: Default::default(),
            #[cfg(feature = "gpu")]
            gpu_terms: None,
        })
    }

    fn tick(&self, name: &'static str, since: Instant) {
        *self.timers.borrow_mut().entry(name).or_default() += since.elapsed().as_secs_f64();
    }

    fn local(&self) -> usize {
        self.g.batch
    }

    /// Rows `[local, w]` of this group's members -> rows `[members, w]` of all.
    fn gather_rows(&self, local: &[f64], w: usize) -> Vec<f64> {
        let mut all = vec![0.0; self.members * w];
        all[self.first_member * w..self.first_member * w + local.len()].copy_from_slice(local);
        if let Some(c) = self.cross {
            c.allreduce_sum(&mut all);
        }
        all
    }

    fn gather_complex(&self, local: &[C64]) -> Vec<C64> {
        let flat: Vec<f64> = local.iter().flat_map(|c| [c.re, c.im]).collect();
        self.gather_rows(&flat, 2 * self.plan.len()).chunks(2).map(|p| C64::new(p[0], p[1])).collect()
    }

    /// Whole maps `[local, npix]` from the owned slices `[local, owned]`
    /// (`x` itself on one rank).
    pub fn whole<'x>(&self, x: &'x [f64]) -> Cow<'x, [f64]> {
        let (s, e) = self.g.owned();
        let (s, e) = (s as usize, e as usize);
        if self.g.comm.size() == 1 {
            return Cow::Borrowed(x);
        }
        let mut full = vec![0.0; self.local() * self.npix];
        for (m, row) in x.chunks(e - s).enumerate() {
            full[m * self.npix + s..m * self.npix + e].copy_from_slice(row);
        }
        self.g.comm.allreduce_sum(&mut full);
        Cow::Owned(full)
    }

    /// Owned slices of whole maps (`full` itself on one rank).
    fn owned<'f>(&self, full: &'f [f64]) -> Cow<'f, [f64]> {
        let (s, e) = self.g.owned();
        if self.g.comm.size() == 1 {
            return Cow::Borrowed(full);
        }
        Cow::Owned(full.chunks(self.npix).flat_map(|row| row[s as usize..e as usize].iter().copied()).collect())
    }

    /// `kappa = f(g) - mean(f(g))` and `f'(g)` of whole maps: the pixels on
    /// the host threads, each member's mean summed in pixel order.
    pub fn physical(&self, g: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (mut k, mut d) = (vec![0.0; g.len()], vec![0.0; g.len()]);
        par_zip(&mut k, &mut d, |lo, ks, ds| {
            for (i, (a, b)) in ks.iter_mut().zip(ds.iter_mut()).enumerate() {
                (*a, *b) = self.cal.lut.inverse(g[lo + i]);
            }
        });
        let means: Vec<f64> = k.chunks(self.npix).map(|row| row.iter().sum::<f64>() / self.npix as f64).collect();
        let npix = self.npix;
        par_chunks(&mut k, |lo, ks| ks.iter_mut().enumerate().for_each(|(i, v)| *v -= means[(lo + i) / npix]));
        (k, d)
    }

    /// One SC block through `obj` on the owned slices `x`: the loss and the
    /// owned gradient.
    fn sc_block(&self, obj: &Objective<'a, C>, x: &[f64], physical: bool) -> (Block, Vec<f64>) {
        let n = self.plan.len();
        let t = &self.cal.targets;
        let (mean, var, sigma) = if physical { (&t.psc_mean, &t.psc_var, &t.psc_sigma) } else { (&t.gsc_mean, &t.gsc_var, &t.gsc_sigma) };
        let mut block = Block::default();
        let (_, grad) = obj.loss_and_grad_with(x, |raw| {
            let all = self.gather_complex(raw);
            let feats = self.coords.forward(&all, n).expect("SC coordinates");
            let (b, cot) = sc_loss(&feats, mean, var, sigma, self.plan, physical).expect("SC block");
            block = b;
            let raw_cot = self.coords.vjp(&all, &cot, n).expect("SC coordinates");
            let off = self.first_member * n;
            (b.loss, raw_cot[off..off + raw.len()].to_vec())
        });
        (block, grad)
    }

    #[cfg(feature = "gpu")]
    fn add_time(&self, name: &'static str, seconds: f64) {
        *self.timers.borrow_mut().entry(name).or_default() += seconds;
    }

    /// Band powers, CIC and Minkowski functionals of the members' whole
    /// physical maps on the host, and what their cotangents need.
    fn host_forward(&self, kfull: &[f64], kept: &mut HostPass) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let t = Instant::now();
        let local = self.local();
        let mut pw = Vec::with_capacity(local * self.windows.len());
        let mut hist = Vec::with_capacity(local * self.cic.bins);
        for row in kfull.chunks(self.npix) {
            let (p, a) = band_powers(&self.sht, &self.windows, row);
            pw.extend(p);
            kept.alms.push(a);
            hist.extend(self.cic.forward(row));
        }
        self.tick("power_pdf_forward", t);
        // Minkowski states (six maps per member) kept for the cotangent when
        // they take at most MF_STATE_BYTES, else recomputed (the source's replay).
        let t = Instant::now();
        let keep = local * self.npix * 6 * 8 <= MF_STATE_BYTES;
        let mut mfv = Vec::with_capacity(local * self.mk.count());
        for row in kfull.chunks(self.npix) {
            let (v, s) = self.mk.forward(&self.sht, row);
            mfv.extend(v);
            if keep {
                kept.states.push(s);
            }
        }
        self.tick("mf_forward", t);
        (pw, hist, mfv)
    }

    /// Map cotangents `[local, npix]` on the host from the cotangents of this
    /// group's members' band powers, CIC and functionals.
    fn host_reverse(&self, kfull: &[f64], kept: HostPass, cps: &[f64], cpdf: &[f64], cmf: &[f64]) -> Vec<f64> {
        let (nb, bins, nm) = (self.windows.len(), self.cic.bins, self.mk.count());
        let t = Instant::now();
        let mut hk = vec![0.0; self.local() * self.npix];
        for (m, row) in kfull.chunks(self.npix).enumerate() {
            let g1 = band_powers_vjp(&self.sht, &self.windows, &kept.alms[m], &cps[m * nb..(m + 1) * nb]);
            let g2 = self.cic.vjp(row, &cpdf[m * bins..(m + 1) * bins]);
            let out = &mut hk[m * self.npix..(m + 1) * self.npix];
            out.iter_mut().zip(g1.iter().zip(&g2)).for_each(|(o, (p, q))| *o = p + q);
        }
        self.tick("power_pdf_reverse", t);
        let t = Instant::now();
        let mut states = kept.states.into_iter();
        for (m, row) in kfull.chunks(self.npix).enumerate() {
            let state = match states.next() {
                Some(s) => s,
                None => self.mk.forward(&self.sht, row).1,
            };
            let g3 = self.mk.vjp(&self.sht, &state, &cmf[m * nm..(m + 1) * nm]);
            hk[m * self.npix..(m + 1) * self.npix].iter_mut().zip(g3).for_each(|(o, v)| *o += v);
        }
        self.tick("mf_reverse", t);
        hk
    }

    /// Runs the harmonic terms on `gpu` (`--harmonic gpu`).
    #[cfg(feature = "gpu")]
    pub fn use_device_terms(&mut self, gpu: std::sync::Arc<stl_gpu::Gpu>) -> Result<(), String> {
        self.gpu_terms = Some(RefCell::new(GpuTerms::new(gpu, &self.sht, &self.mk)?));
        Ok(())
    }

    /// Whether the harmonic terms run on the device.
    pub fn device_terms(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.gpu_terms.is_some();
        #[cfg(not(feature = "gpu"))]
        false
    }

    #[cfg(feature = "gpu")]
    fn device_forward(&self, kfull: &[f64]) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let d = self.gpu_terms.as_ref()?;
        let (pw, hist, mfv, secs) = d.borrow_mut().forward(kfull, &self.windows, &self.cic);
        self.add_time("power_pdf_forward", secs[0]);
        self.add_time("mf_forward", secs[1]);
        Some((pw, hist, mfv))
    }

    #[cfg(not(feature = "gpu"))]
    fn device_forward(&self, _: &[f64]) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        None
    }

    #[cfg(feature = "gpu")]
    fn device_reverse(&self, kfull: &[f64], cps: &[f64], cpdf: &[f64], cmf: &[f64]) -> Option<Vec<f64>> {
        let d = self.gpu_terms.as_ref()?;
        let (hk, secs) = d.borrow_mut().reverse(kfull, &self.windows, &self.cic, cps, cpdf, cmf);
        self.add_time("power_pdf_reverse", secs[0]);
        self.add_time("mf_reverse", secs[1]);
        Some(hk)
    }

    #[cfg(not(feature = "gpu"))]
    fn device_reverse(&self, _: &[f64], _: &[f64], _: &[f64], _: &[f64]) -> Option<Vec<f64>> {
        None
    }

    /// Loss and owned gradient of this group's members `x` `[local, owned]`
    /// (Gaussian maps).
    pub fn loss_and_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let t0 = Instant::now();
        let gfull = self.whole(x);
        let (kfull, slope) = self.physical(&gfull);
        let kx = self.owned(&kfull);
        self.tick("host_physical_map", t0);
        let t = Instant::now();
        let (gsc, mut grad) = self.sc_block(self.g, x, false);
        self.tick("gaussian_sc", t);
        let t = Instant::now();
        let (psc, mut gk) = self.sc_block(self.k, &kx, true);
        self.tick("physical_sc", t);
        // Harmonic features of this group's members (on the device with
        // `--harmonic gpu`), gathered over the groups.
        let local = self.local();
        let (nb, bins, nm) = (self.windows.len(), self.cic.bins, self.mk.count());
        let mut kept = HostPass::default();
        let (pw, hist, mfv) = match self.device_forward(&kfull) {
            Some(f) => f,
            None => self.host_forward(&kfull, &mut kept),
        };
        let tg = &self.cal.targets;
        let (ps, cps) = power_loss(&self.gather_rows(&pw, nb), &tg.power_mean, &tg.power_var, &self.centers, self.cutoff).expect("C_l block");
        let (pdf, cpdf) = pdf_loss(&self.gather_rows(&hist, bins), &tg.pdf_mean, &tg.pdf_var);
        let (mf, cmf) = mf_loss(&self.gather_rows(&mfv, nm), &tg.mf_mean, &tg.mf_var);
        let (a, b) = (self.first_member, self.first_member + local);
        let (cps, cpdf, cmf) = (&cps[a * nb..b * nb], &cpdf[a * bins..b * bins], &cmf[a * nm..b * nm]);
        let hk = match self.device_reverse(&kfull, cps, cpdf, cmf) {
            Some(h) => h,
            None => self.host_reverse(&kfull, kept, cps, cpdf, cmf),
        };
        // kappa cotangent -> g: f'(g) (c - mean(c)), member by member.
        let t = Instant::now();
        let hk = self.owned(&hk);
        par_chunks(&mut gk, |lo, gs| gs.iter_mut().enumerate().for_each(|(i, a)| *a += hk[lo + i]));
        let owned = gk.len() / local.max(1);
        let sums: Vec<f64> = gk.chunks(owned.max(1)).map(|r| r.iter().sum()).collect();
        let mut sums = sums;
        self.g.comm.allreduce_sum(&mut sums);
        let sl = self.owned(&slope);
        let npix = self.npix as f64;
        par_chunks(&mut grad, |lo, gs| {
            for (i, v) in gs.iter_mut().enumerate() {
                let p = lo + i;
                *v = (*v + sl[p] * (gk[p] - sums[p / owned.max(1)] / npix)) / 5.0;
            }
        });
        self.tick("host_chain", t);
        let blocks = [gsc, psc, ps, pdf, mf];
        *self.last.borrow_mut() = blocks;
        self.tick("evaluation", t0);
        (blocks.iter().map(|b| b.loss).sum::<f64>() / 5.0, grad)
    }
}
