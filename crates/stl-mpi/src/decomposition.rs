//! Contiguous NESTED ranges per rank and the cells each rank must hold.

use std::collections::BTreeSet;
use stl_core::engine::{block_domains, decimation_domain, supports, EngineConfig, Layout, Overlap, Partition};
use stl_core::local::LocalGeometry;
use stl_core::neighbours::{children, dilate_fine, parents};

/// Level-0 ranges `[starts[r], starts[r + 1])`, multiples of `unit`
/// (`4^k`, `k <= levels - 1` as large as the number of ranks allows).  A
/// rank owns the cell `c` of level `d` when its first descendant at level
/// 0, `c 4^d`, is in its range: `[ceil(starts[r] / 4^d), ceil(starts[r+1] /
/// 4^d))`.  When `unit = 4^(levels-1)` the pyramid of every rank is aligned
/// (it owns the parents of all its cells); with more ranks than cells at
/// the coarsest levels, some ranks own no cell there and a parent may be
/// owned by the neighbour of its children's owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decomposition {
    pub levels: usize,
    pub starts: Vec<i64>,
    pub unit: i64,
}

impl Decomposition {
    /// Balanced split of the sky at `nside0` over `size` ranks.
    pub fn new(geo: &LocalGeometry, levels: usize, size: usize) -> Self {
        let npix0 = geo.pixels(0) as i64;
        assert!(size as i64 <= npix0, "{size} ranks for {npix0} pixels");
        let mut k = levels - 1;
        while k > 0 && npix0 >> (2 * k) < size as i64 {
            k -= 1;
        }
        Self::with_unit(geo, levels, size, 1 << (2 * k))
    }

    /// Split aligned on `unit` (a power of 4 dividing the sky).
    pub fn with_unit(geo: &LocalGeometry, levels: usize, size: usize, unit: i64) -> Self {
        let npix0 = geo.pixels(0) as i64;
        assert!(unit > 0 && unit & (unit - 1) == 0 && unit.trailing_zeros().is_multiple_of(2), "unit must be a power of 4");
        let n_units = npix0 / unit;
        assert!(size as i64 <= n_units, "{size} ranks for {n_units} units of {unit} pixels");
        let starts = (0..=size as i64).map(|r| r * n_units / size as i64 * unit).collect();
        Decomposition { levels, starts, unit }
    }

    pub fn size(&self) -> usize {
        self.starts.len() - 1
    }

    /// `ceil(v / 4^level)`.
    fn coarse(v: i64, level: usize) -> i64 {
        let s = 2 * level as u32;
        (v + (1 << s) - 1) >> s
    }

    /// Owned range of `rank` at `level` (possibly empty).
    pub fn owned(&self, rank: usize, level: usize) -> (i64, i64) {
        (Self::coarse(self.starts[rank], level), Self::coarse(self.starts[rank + 1], level))
    }

    /// Rank owning cell `id` of `level`.
    pub fn owner(&self, level: usize, id: i64) -> usize {
        let first = id << (2 * level);
        self.starts[1..].partition_point(|&s| s <= first)
    }
}

/// Blocks, owned ranges and layouts of `rank`.  A level's layout is the
/// owned range plus every cell the rank's blocks touch at that level:
///
/// * blocks of the level: their convolution domain (block + two
///   convolution halos: maps read, map cotangents written, `a1` moduli read
///   and their cotangents written) and their decimation domain (block +
///   two decimation halos, read by the transposed decimation, which reads
///   the parents of that domain at the next level);
/// * blocks of the next level: the decimation domain of their children
///   (maps read by `decimate`);
/// * the parents of the owned cells (band cotangents reduced to them);
/// * blocks of every finer level: the supports of their decimation chain
///   at this level (stored moduli written, their cotangents read).
pub fn build_partition(geo: &LocalGeometry, dec: &Decomposition, rank: usize) -> Partition {
    let levels = dec.levels;
    let owned: Vec<(i64, i64)> = (0..levels).map(|d| dec.owned(rank, d)).collect();
    let blocks: Vec<Vec<Vec<i64>>> = owned.iter().map(|&(s, e)| chunk(&(s..e).collect::<Vec<_>>(), geo.block_pixels)).collect();
    let mut held: Vec<BTreeSet<i64>> = owned.iter().map(|&(s, e)| (s..e).collect()).collect();
    for (n, level_blocks) in blocks.iter().enumerate() {
        for block in level_blocks {
            hold_block(geo, levels, n, block, &mut held);
        }
    }
    let layouts = held.into_iter().map(|s| Layout::Sparse(s.into_iter().collect())).collect();
    Partition {
        owned,
        layouts,
        halo_blocks: vec![Vec::new(); levels],
        map_blocks: blocks.clone(),
        blocks,
        overlap: None,
    }
}

/// [`build_partition`] or [`build_overlap_partition`] as `cfg.overlap` asks.
pub fn build_partition_for(geo: &LocalGeometry, cfg: &EngineConfig, dec: &Decomposition, rank: usize) -> Partition {
    match cfg.overlap {
        None => build_partition(geo, dec, rank),
        Some(coarse) => build_overlap_partition(geo, dec, rank, coarse),
    }
}

/// `ids` (sorted) cut into blocks of `size` cells.
fn chunk(ids: &[i64], size: usize) -> Vec<Vec<i64>> {
    ids.chunks(size.max(1)).map(|c| c.to_vec()).collect()
}

fn union(a: &[i64], b: &[i64]) -> Vec<i64> {
    let s: BTreeSet<i64> = a.iter().chain(b).copied().collect();
    s.into_iter().collect()
}

/// Cells a block of level `n` touches (see [`build_partition`]).
fn hold_block(geo: &LocalGeometry, levels: usize, n: usize, block: &[i64], held: &mut [BTreeSet<i64>]) {
    let (_, domain) = block_domains(geo, n, block);
    held[n].extend(domain);
    if n + 1 < levels {
        let down = dilate_fine(block, geo.nside(n), 2 * geo.down_rings(n));
        held[n + 1].extend(parents(&down));
        held[n].extend(down);
        held[n + 1].extend(parents(block));
    }
    if n > 0 {
        held[n - 1].extend(decimation_domain(geo, n - 1, block));
    }
    for (k, support) in supports(geo, levels, n, block).into_iter().enumerate().skip(1) {
        held[n + k].extend(support);
    }
}

/// Partition of the overlapping-patch mode ([`Overlap`]), levels from
/// `coarse` on held whole and evaluated by rank 0.
///
/// Work regions, from the last fine level down: `W_k = owned_k U S_k` with
/// `S_k = pre_k(S_{k+1} U dilate(W_{k+1}, conv))`, `pre_k(T)` the cells of
/// level `k` whose decimation chain reaches `T` (children of `T` dilated by
/// the decimation rings): every stored modulus a block of `W_{k+1}` reads
/// on its convolution rows then gets all its pieces locally.  Map regions:
/// `M_k = dilate(W_k, 2 conv) U` the decimation domain of `M_{k+1}` (of the
/// owned cells of the first coarse level for the last fine one).
pub fn build_overlap_partition(geo: &LocalGeometry, dec: &Decomposition, rank: usize, coarse: usize) -> Partition {
    let levels = dec.levels;
    assert!(coarse >= 1 && coarse <= levels, "first coarse level {coarse} out of 1..={levels}");
    let root = rank == 0;
    let owned: Vec<(i64, i64)> = (0..levels).map(|d| dec.owned(rank, d)).collect();
    let owned_ids: Vec<Vec<i64>> = owned.iter().map(|&(s, e)| (s..e).collect()).collect();
    let c = geo.conv_rings;
    let pre = |k: usize, t: &[i64]| dilate_fine(&children(t), geo.nside(k), geo.down_rings(k));
    // Work regions.
    let mut work: Vec<Vec<i64>> = vec![Vec::new(); levels];
    let mut s_next: Vec<i64> = Vec::new();
    for k in (0..coarse).rev() {
        work[k] = union(&owned_ids[k], &s_next);
        if k > 0 {
            let need = union(&s_next, &dilate_fine(&work[k], geo.nside(k), c));
            s_next = pre(k - 1, &need);
        }
    }
    // Map regions.
    let mut maps: Vec<Vec<i64>> = vec![Vec::new(); levels];
    let mut m_next: Vec<i64> = if coarse < levels { owned_ids[coarse].clone() } else { Vec::new() };
    for k in (0..coarse).rev() {
        let mut mk = dilate_fine(&dilate_fine(&work[k], geo.nside(k), c), geo.nside(k), c);
        if !m_next.is_empty() {
            mk = union(&mk, &dilate_fine(&children(&m_next), geo.nside(k), 2 * geo.down_rings(k)));
        }
        maps[k] = mk.clone();
        m_next = mk;
    }
    let all = |k: usize| -> Vec<i64> { (0..geo.pixels(k) as i64).collect() };
    let mut blocks = Vec::with_capacity(levels);
    let mut halo_blocks = Vec::with_capacity(levels);
    let mut map_blocks = Vec::with_capacity(levels);
    for k in 0..levels {
        if k < coarse {
            blocks.push(chunk(&owned_ids[k], geo.block_pixels));
            let extra: Vec<i64> = {
                let o: BTreeSet<i64> = owned_ids[k].iter().copied().collect();
                work[k].iter().copied().filter(|v| !o.contains(v)).collect()
            };
            halo_blocks.push(chunk(&extra, geo.block_pixels));
            map_blocks.push(chunk(&maps[k], geo.block_pixels));
        } else {
            blocks.push(if root { chunk(&all(k), geo.block_pixels) } else { Vec::new() });
            halo_blocks.push(Vec::new());
            map_blocks.push(if k == coarse {
                chunk(&owned_ids[k], geo.block_pixels)
            } else if root {
                chunk(&all(k), geo.block_pixels)
            } else {
                Vec::new()
            });
        }
    }
    // Layouts: the fine levels hold what their work and map blocks touch.
    let mut held: Vec<BTreeSet<i64>> = owned_ids.iter().map(|ids| ids.iter().copied().collect()).collect();
    for k in 0..coarse {
        held[k].extend(maps[k].iter().copied());
        for block in blocks[k].iter().chain(&halo_blocks[k]).chain(&map_blocks[k]) {
            hold_block(geo, levels, k, block, &mut held);
        }
    }
    let layouts = (0..levels)
        .map(|k| if k < coarse { Layout::Sparse(std::mem::take(&mut held[k]).into_iter().collect()) } else { Layout::Full(geo.pixels(k)) })
        .collect();
    Partition { owned, layouts, blocks, halo_blocks, map_blocks, overlap: Some(Overlap { coarse, root }) }
}
