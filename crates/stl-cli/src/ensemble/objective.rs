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
use super::fields::{band_powers, band_powers_vjp, lp_windows, Cic, Minkowski};
use super::losses::{mf_loss, pdf_loss, power_loss, sc_loss, Block, ScCoordinates};
use num_complex::Complex64;
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

    /// Whole maps `[local, npix]` from the owned slices `[local, owned]`.
    pub fn whole(&self, x: &[f64]) -> Vec<f64> {
        let (s, e) = self.g.owned();
        let (s, e) = (s as usize, e as usize);
        if self.g.comm.size() == 1 {
            return x.to_vec();
        }
        let mut full = vec![0.0; self.local() * self.npix];
        for (m, row) in x.chunks(e - s).enumerate() {
            full[m * self.npix + s..m * self.npix + e].copy_from_slice(row);
        }
        self.g.comm.allreduce_sum(&mut full);
        full
    }

    fn owned(&self, full: &[f64]) -> Vec<f64> {
        let (s, e) = self.g.owned();
        full.chunks(self.npix).flat_map(|row| row[s as usize..e as usize].iter().copied()).collect()
    }

    /// `kappa = f(g) - mean(f(g))` and `f'(g)` of whole maps.
    pub fn physical(&self, g: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut k = Vec::with_capacity(g.len());
        let mut d = Vec::with_capacity(g.len());
        for row in g.chunks(self.npix) {
            let start = k.len();
            for &v in row {
                let (a, b) = self.cal.lut.inverse(v);
                k.push(a);
                d.push(b);
            }
            let mean = k[start..].iter().sum::<f64>() / self.npix as f64;
            k[start..].iter_mut().for_each(|v| *v -= mean);
        }
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
        // Host features of this group's members, gathered over the groups.
        let t = Instant::now();
        let local = self.local();
        let nb = self.windows.len();
        let nm = self.mk.count();
        let mut pw = Vec::with_capacity(local * nb);
        let mut alms = Vec::with_capacity(local);
        let mut hist = Vec::with_capacity(local * self.cic.bins);
        for row in kfull.chunks(self.npix) {
            let (p, a) = band_powers(&self.sht, &self.windows, row);
            pw.extend(p);
            alms.push(a);
            hist.extend(self.cic.forward(row));
        }
        self.tick("power_pdf_forward", t);
        // Minkowski states (six maps per member) kept for the cotangent when
        // they take at most MF_STATE_BYTES, else recomputed (the source's replay).
        let t = Instant::now();
        let keep = local * self.npix * 6 * 8 <= MF_STATE_BYTES;
        let mut mfv = Vec::with_capacity(local * nm);
        let mut states = Vec::new();
        for row in kfull.chunks(self.npix) {
            let (v, s) = self.mk.forward(&self.sht, row);
            mfv.extend(v);
            if keep {
                states.push(s);
            }
        }
        self.tick("mf_forward", t);
        let tg = &self.cal.targets;
        let (ps, cps) = power_loss(&self.gather_rows(&pw, nb), &tg.power_mean, &tg.power_var, &self.centers, self.cutoff).expect("C_l block");
        let (pdf, cpdf) = pdf_loss(&self.gather_rows(&hist, self.cic.bins), &tg.pdf_mean, &tg.pdf_var);
        let (mf, cmf) = mf_loss(&self.gather_rows(&mfv, nm), &tg.mf_mean, &tg.mf_var);
        let t = Instant::now();
        let mut hk = vec![0.0; local * self.npix];
        for (m, row) in kfull.chunks(self.npix).enumerate() {
            let a = self.first_member + m;
            let g1 = band_powers_vjp(&self.sht, &self.windows, &alms[m], &cps[a * nb..(a + 1) * nb]);
            let g2 = self.cic.vjp(row, &cpdf[a * self.cic.bins..(a + 1) * self.cic.bins]);
            let out = &mut hk[m * self.npix..(m + 1) * self.npix];
            out.iter_mut().zip(g1.iter().zip(&g2)).for_each(|(o, (p, q))| *o = p + q);
        }
        self.tick("power_pdf_reverse", t);
        let t = Instant::now();
        let mut kept = states.into_iter();
        for (m, row) in kfull.chunks(self.npix).enumerate() {
            let a = self.first_member + m;
            let state = match kept.next() {
                Some(s) => s,
                None => self.mk.forward(&self.sht, row).1,
            };
            let g3 = self.mk.vjp(&self.sht, &state, &cmf[a * nm..(a + 1) * nm]);
            hk[m * self.npix..(m + 1) * self.npix].iter_mut().zip(g3).for_each(|(o, v)| *o += v);
        }
        self.tick("mf_reverse", t);
        // kappa cotangent -> g: f'(g) (c - mean(c)), member by member.
        let t = Instant::now();
        let hk = self.owned(&hk);
        gk.iter_mut().zip(&hk).for_each(|(a, b)| *a += b);
        let owned = gk.len() / local.max(1);
        let sums: Vec<f64> = gk.chunks(owned.max(1)).map(|r| r.iter().sum()).collect();
        let mut sums = sums;
        self.g.comm.allreduce_sum(&mut sums);
        let sl = self.owned(&slope);
        for (m, (row, srow)) in gk.chunks(owned.max(1)).zip(sl.chunks(owned.max(1))).enumerate() {
            let mean = sums[m] / self.npix as f64;
            for (i, (c, d)) in row.iter().zip(srow).enumerate() {
                grad[m * owned + i] += d * (c - mean);
            }
        }
        grad.iter_mut().for_each(|v| *v /= 5.0);
        self.tick("host_chain", t);
        let blocks = [gsc, psc, ps, pdf, mf];
        *self.last.borrow_mut() = blocks;
        self.tick("evaluation", t0);
        (blocks.iter().map(|b| b.loss).sum::<f64>() / 5.0, grad)
    }
}
