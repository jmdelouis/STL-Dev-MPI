//! Contiguous NESTED ranges per rank and the cells each rank must hold.

use std::collections::BTreeSet;
use stl_core::engine::{block_domains, supports, Layout, Partition};
use stl_core::local::LocalGeometry;
use stl_core::neighbours::{dilate_fine, parents};

/// Level-0 ranges `[starts[r], starts[r + 1])`, multiples of
/// `4^(levels - 1)` so that the range of rank `r` at level `d` is
/// `[starts[r] / 4^d, starts[r + 1] / 4^d)`: the pyramid of a rank is
/// aligned and the coarsest level is shared without replication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decomposition {
    pub levels: usize,
    pub starts: Vec<i64>,
}

impl Decomposition {
    /// Balanced split of the sky at `nside0` over `size` ranks.
    pub fn new(geo: &LocalGeometry, levels: usize, size: usize) -> Self {
        let unit = 1i64 << (2 * (levels - 1));
        let n_units = geo.pixels(0) as i64 / unit;
        assert!(
            size as i64 <= n_units,
            "{size} ranks for {n_units} pixels at the coarsest level: use fewer ranks or levels"
        );
        let starts = (0..=size as i64).map(|r| r * n_units / size as i64 * unit).collect();
        Decomposition { levels, starts }
    }

    pub fn size(&self) -> usize {
        self.starts.len() - 1
    }

    /// Owned range of `rank` at `level`.
    pub fn owned(&self, rank: usize, level: usize) -> (i64, i64) {
        let shift = 2 * level;
        (self.starts[rank] >> shift, self.starts[rank + 1] >> shift)
    }

    /// Rank owning cell `id` of `level`.
    pub fn owner(&self, level: usize, id: i64) -> usize {
        let shift = 2 * level;
        self.starts[1..].partition_point(|&s| (s >> shift) <= id)
    }
}

/// Blocks, owned ranges and layouts of `rank`.  A level's layout is the
/// owned range plus every cell the rank's blocks touch at that level:
///
/// * blocks of the level: their convolution domain (block + two
///   convolution halos: maps read, map cotangents written, `a1` moduli read
///   and their cotangents written) and their decimation domain (block +
///   two decimation halos: maps read by `decimate`);
/// * blocks of the finer level: the parents of their decimation domain
///   (cotangents read by the transposed decimation);
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
            }
            for (k, support) in supports(geo, levels, n, block).into_iter().enumerate().skip(1) {
                held[n + k].extend(support);
            }
        }
    }
    let layouts = held.into_iter().map(|s| Layout::Sparse(s.into_iter().collect())).collect();
    Partition { owned, layouts, blocks }
}
