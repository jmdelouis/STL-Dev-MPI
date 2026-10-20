//! What a rank's device engine needs besides the fields, computed once on
//! the host (it depends on the decomposition, not on the map): the cells
//! held per level, the blocks of owned cells with their convolution halo,
//! and the full-sky decimation rows restricted to the owned cells (forward)
//! and to the owned fine cells (transpose).
//!
//! Device blocks are contiguous ranges of owned cells, much larger than the
//! CPU blocks; the layouts of the partition do not depend on the block size
//! (dilations distribute over unions), so the device blocks read the same
//! halo as the CPU ones.

use stl_core::decimation::decimation_row;
use stl_core::engine::{Layout, Partition};
use stl_core::local::LocalGeometry;
use stl_core::neighbours::{dilate, dilate_fine, parents};

/// A block: owned cells `[start, start + len)` of a level and the output
/// rows of its convolution, `a1` = block + one convolution halo.
#[derive(Clone, Debug)]
pub struct BlockTopo {
    pub start: i64,
    pub len: usize,
    pub a1: Vec<i64>,
    /// Positions of `a1` in the level's layout.
    pub a1pos: Vec<u32>,
    /// Position of `start` in `a1` (the block is contiguous there).
    pub f0: usize,
}

#[derive(Clone, Debug)]
pub struct LevelTopo {
    pub nside: i64,
    /// Held cells (the layout), sorted.
    pub ids: Vec<i64>,
    pub owned: (i64, i64),
    /// Position of the first owned cell in the layout.
    pub own_pos0: usize,
    pub blocks: Vec<BlockTopo>,
}

impl LevelTopo {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn owned_len(&self) -> usize {
        (self.owned.1 - self.owned.0) as usize
    }
}

/// Sparse rows with `u32` columns.
#[derive(Clone, Debug, Default)]
pub struct HostCsr {
    pub indptr: Vec<u32>,
    pub cols: Vec<u32>,
    pub vals: Vec<f64>,
}

impl HostCsr {
    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }
}

pub struct Topology {
    pub levels: Vec<LevelTopo>,
    /// `down[d]`: rows = owned cells of level `d + 1`, columns = layout positions of level `d`.
    pub down: Vec<HostCsr>,
    /// `up[d]`: transpose, rows = owned cells of level `d`, columns = layout positions of level `d + 1`,
    /// coarse cells in increasing order within a row.
    pub up: Vec<HostCsr>,
}

fn layout_ids(layout: &Layout) -> Vec<i64> {
    match layout {
        Layout::Full(n) => (0..*n as i64).collect(),
        Layout::Sparse(ids) => ids.clone(),
    }
}

/// Position of `id` in the sorted `ids`.
fn find(ids: &[i64], id: i64, what: &str) -> u32 {
    let q = ids.partition_point(|&s| s < id);
    assert!(q < ids.len() && ids[q] == id, "{what}: cell {id} is not held by this rank");
    q as u32
}

/// `f` over chunks of `items` on `threads` host threads, results in order.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&[T]) -> Vec<R> + Sync) -> Vec<R> {
    let threads = threads.max(1);
    if items.len() < 1024 || threads == 1 {
        return f(items);
    }
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = items.chunks(chunk).map(|c| s.spawn(|| f(c))).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// [`dilate_fine`] of a large sorted set, by chunks on several threads.
pub fn par_dilate_fine(ids: &[i64], nside: i64, rings: u32, threads: usize) -> Vec<i64> {
    let mut out = par_map(ids, threads, |c| dilate_fine(c, nside, rings));
    out.sort_unstable();
    out.dedup();
    out
}

impl Topology {
    /// Topology of the rank holding `part`, device blocks of at most
    /// `block_rows` owned cells.
    pub fn new(geo: &LocalGeometry, part: &Partition, block_rows: usize, threads: usize) -> Self {
        let levels = part.layouts.len();
        let mut lv = Vec::with_capacity(levels);
        for n in 0..levels {
            let ids = layout_ids(&part.layouts[n]);
            let nside = geo.nside(n);
            let (s, e) = part.owned[n];
            let own_pos0 = if s < e { find(&ids, s, "owned") as usize } else { 0 };
            let mut blocks = Vec::new();
            let mut b = s;
            while b < e {
                let len = (block_rows as i64).min(e - b) as usize;
                let cells: Vec<i64> = (b..b + len as i64).collect();
                let a1 = par_dilate_fine(&cells, nside, geo.conv_rings, threads);
                let a1pos: Vec<u32> = par_map(&a1, threads, |c| c.iter().map(|&v| find(&ids, v, "a1")).collect());
                let f0 = a1.partition_point(|&v| v < b);
                assert_eq!(a1[f0 + len - 1], b + len as i64 - 1);
                blocks.push(BlockTopo { start: b, len, a1, a1pos, f0 });
                b += len as i64;
            }
            lv.push(LevelTopo { nside, ids, owned: (s, e), own_pos0, blocks });
        }
        let mut down = Vec::new();
        let mut up = Vec::new();
        for d in 0..levels.saturating_sub(1) {
            let (dn, upn) = decimation(geo, &lv[d], &lv[d + 1], threads);
            down.push(dn);
            up.push(upn);
        }
        Topology { levels: lv, down, up }
    }
}

/// Decimation rows from `fine` (level `d`) to `coarse` (level `d + 1`).
fn decimation(geo: &LocalGeometry, fine: &LevelTopo, coarse: &LevelTopo, threads: usize) -> (HostCsr, HostCsr) {
    let nside = fine.nside;
    let params = geo.down_params;
    // Forward: owned coarse cells.
    let owned_c: Vec<i64> = (coarse.owned.0..coarse.owned.1).collect();
    let rows: Vec<(Vec<u32>, Vec<f64>)> = par_map(&owned_c, threads, |cs| {
        cs.iter()
            .map(|&c| {
                let (f, w) = decimation_row(c, nside, &params);
                (f.iter().map(|&v| find(&fine.ids, v, "decimation")).collect(), w)
            })
            .collect()
    });
    let mut dn = HostCsr { indptr: vec![0], ..Default::default() };
    for (cols, vals) in rows {
        dn.cols.extend(cols);
        dn.vals.extend(vals);
        dn.indptr.push(dn.cols.len() as u32);
    }
    // Transpose on the owned fine cells: every coarse cell whose candidates
    // may contain one (3 coarse rings around their parents, a superset).
    let (fs, fe) = fine.owned;
    let mut up = HostCsr { indptr: vec![0], ..Default::default() };
    if fs < fe {
        let fine_owned: Vec<i64> = (fs..fe).collect();
        let near = par_map(&parents(&fine_owned), threads, |p| dilate(p, nside / 2, 3));
        let mut near = near;
        near.sort_unstable();
        near.dedup();
        // (fine, coarse, weight) for the owned fine cells.
        let mut entries: Vec<(i64, i64, f64)> = par_map(&near, threads, |cs| {
            let mut out = Vec::new();
            for &c in cs {
                let (f, w) = decimation_row(c, nside, &params);
                for (fi, wi) in f.into_iter().zip(w) {
                    if fi >= fs && fi < fe {
                        out.push((fi, c, wi));
                    }
                }
            }
            out
        });
        entries.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut k = 0;
        for j in fs..fe {
            while k < entries.len() && entries[k].0 == j {
                up.cols.push(find(&coarse.ids, entries[k].1, "decimation transpose"));
                up.vals.push(entries[k].2);
                k += 1;
            }
            up.indptr.push(up.cols.len() as u32);
        }
    }
    (dn, up)
}
