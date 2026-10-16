//! Contiguous NESTED ranges per rank and the cells each rank must hold.

use std::collections::BTreeSet;
use stl_core::engine::{block_domains, decimation_domain, supports, Layout, Partition};
use stl_core::local::LocalGeometry;
use stl_core::neighbours::{dilate_fine, parents};

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
    let blocks: Vec<Vec<Vec<i64>>> = owned
        .iter()
        .map(|&(s, e)| {
            let size = geo.block_pixels as i64;
            (s..e).step_by(geo.block_pixels).map(|b| (b..(b + size).min(e)).collect()).collect()
        })
        .collect();
    let mut held: Vec<BTreeSet<i64>> = owned.iter().map(|&(s, e)| (s..e).collect()).collect();
    for n in 0..levels {
        for block in &blocks[n] {
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
    }
    let layouts = held.into_iter().map(|s| Layout::Sparse(s.into_iter().collect())).collect();
    Partition { owned, layouts, blocks }
}
