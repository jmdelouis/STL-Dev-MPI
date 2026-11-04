//! `--init-noise conditional` (`stl_cli::conditional`): draws independent of
//! the number of ranks, and conditional on the parent's bin.

use stl_cli::conditional::Conditional;
use stl_cli::pixel_normal;
use stl_mpi::threads::run_ranks;
use stl_mpi::Comm;

const NSIDE: i64 = 16;
const FACTOR: usize = 4;

fn parents() -> (Vec<f64>, f64) {
    let n_c = 12 * (NSIDE / 2 * NSIDE / 2) as usize;
    let p: Vec<f64> = (0..n_c).map(|p| pixel_normal(5, p as i64)).collect();
    let mut s = p.clone();
    s.sort_by(f64::total_cmp);
    let median = 0.5 * (s[n_c / 2 - 1] + s[n_c / 2]);
    (p, median)
}

/// Target: parents from noise; children of parents above their median get
/// details +-1 (alternating), those below get none.
fn target() -> (Vec<f64>, Vec<f64>) {
    let (parents, median) = parents();
    let t = (0..parents.len() * FACTOR).map(|i| {
        let p = parents[i / FACTOR];
        p + if p > median { if i % 2 == 0 { 1.0 } else { -1.0 } } else { 0.0 }
    });
    (t.collect(), parents)
}

fn draw(ranks: usize) -> Vec<f64> {
    let (t, parents) = target();
    let n = t.len();
    run_ranks(ranks, |comm| {
        let (s, e) = (comm.rank() * n / ranks, (comm.rank() + 1) * n / ranks);
        let cond = Conditional::new(comm, &t[s..e], s, FACTOR, NSIDE / 2, 2);
        cond.draw(&parents, s, e, 42)
    })
    .concat()
}

#[test]
fn same_draws_on_one_and_two_ranks() {
    assert_eq!(draw(1), draw(2));
}

#[test]
fn draws_follow_the_details_of_their_bin() {
    let (parents, median) = parents();
    let x = draw(1);
    let w = 2.0 / stl_cli::conditional::HIST as f64 * 2.0;
    let (mut zero, mut one) = (0, 0);
    for (i, v) in x.iter().enumerate() {
        let d = v - parents[i / FACTOR];
        if parents[i / FACTOR] < median - 0.01 {
            assert!(d.abs() <= w, "pixel {i}: detail {d} for a parent below the median");
            zero += 1;
        } else if parents[i / FACTOR] > median + 0.01 {
            assert!((d.abs() - 1.0).abs() <= w, "pixel {i}: detail {d} for a parent above the median");
            one += 1;
        }
    }
    assert!(zero > 1000 && one > 1000);
}

#[test]
fn pooled_copies_draw_as_one_map() {
    let (t, parents) = target();
    let n = t.len();
    let one = run_ranks(2, |comm| {
        let (s, e) = (comm.rank() * n / 2, (comm.rank() + 1) * n / 2);
        let pooled = Conditional::pooled(comm, &[&t[s..e], &t[s..e]], None, s, FACTOR, NSIDE / 2, 2);
        pooled.draw(&parents, s, e, 42)
    })
    .concat();
    assert_eq!(one, draw(1));
}

#[test]
fn harmonic_upsampling_keeps_low_multipoles() {
    // A constant plus a dipole along z at nside 8, upsampled to 16.
    let theta = |nside: i64, p: usize| stl_core::healpix::pix2ang_nest(p as i64, nside).0;
    let f = |nside: i64| -> Vec<f64> { (0..12 * (nside * nside) as usize).map(|p| 2.0 + theta(nside, p).cos()).collect() };
    let up = stl_cli::conditional::upsample(&f(8), 8, 16);
    let want = f(16);
    let err = up.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(err < 1e-3, "max error {err}");
}
