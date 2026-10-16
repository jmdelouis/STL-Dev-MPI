//! Neighbourhoods and halo dilations (`LocalGeometry.dilate`,
//! `dilate_fine`, `parents`, `children`), through `cdshealpix` — the crate
//! behind `healpix_geo.nested.kth_neighbourhood`.

use crate::healpix::{npix, order_of};

/// Cells within `k` rings of `id` at `nside` (cdshealpix order: the cell
/// itself first, then ring after ring).  Requires `k <= nside`.
pub fn kth_neighbourhood(id: i64, nside: i64, k: u32) -> Vec<i64> {
    let layer = cdshealpix::nested::get(order_of(nside) as u8);
    layer.kth_neighbourhood(id as u64, k).into_iter().map(|v| v as i64).collect()
}

/// Sorted unique parents of `ids`.
pub fn parents(ids: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = ids.iter().map(|&v| v.div_euclid(4)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Children of sorted parents, sorted.
pub fn children(ids: &[i64]) -> Vec<i64> {
    ids.iter().flat_map(|&v| (0..4).map(move |c| 4 * v + c)).collect()
}

/// Sorted union of the `rings`-th neighbourhoods of `ids` at `nside`.
pub fn dilate(ids: &[i64], nside: i64, rings: u32) -> Vec<i64> {
    if rings == 0 {
        return ids.to_vec();
    }
    if rings as i64 >= nside {
        // kth_neighbourhood cannot cross a base cell twice; such a halo
        // covers the (small) level anyway.
        return (0..npix(nside)).collect();
    }
    let mut out: Vec<i64> = ids.iter().flat_map(|&v| kth_neighbourhood(v, nside, rings)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Superset of the `rings`-pixel neighbourhood, built at the parent level.
pub fn dilate_fine(ids: &[i64], nside: i64, rings: u32) -> Vec<i64> {
    if nside < 4 || rings <= 1 {
        return dilate(ids, nside, rings);
    }
    children(&dilate(&parents(ids), nside / 2, rings / 2 + 1))
}
