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
    /// First owned cell (NESTED id).
    pub start: i64,
    /// Number of owned cells.
    pub len: usize,
    /// NESTED ids of the convolution rows, sorted.
    pub a1: Vec<i64>,
    /// Positions of `a1` in the level's layout.
    pub a1pos: Vec<u32>,
    /// Position of `start` in `a1` (the block is contiguous there).
    pub f0: usize,
}

/// The topology of one level of the rank: held cells, owned range and blocks.
#[derive(Clone, Debug)]
pub struct LevelTopo {
    /// Resolution of the level.
    pub nside: i64,
    /// Held cells (the layout), sorted.
    pub ids: Vec<i64>,
    /// Owned cells `[owned.0, owned.1)`.
    pub owned: (i64, i64),
    /// Position of the first owned cell in the layout.
    pub own_pos0: usize,
    /// The device blocks of the owned range, in increasing order.
    pub blocks: Vec<BlockTopo>,
}

impl LevelTopo {
    /// Number of held cells.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no cell is held.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Number of owned cells.
    pub fn owned_len(&self) -> usize {
        (self.owned.1 - self.owned.0) as usize
    }
}

/// Sparse rows with `u32` columns.
#[derive(Clone, Debug, Default)]
pub struct HostCsr {
    /// Row starts in `cols` / `vals` (`n_rows + 1`).
    pub indptr: Vec<u32>,
    /// Column of every nonzero.
    pub cols: Vec<u32>,
    /// Value of every nonzero.
    pub vals: Vec<f64>,
}

impl HostCsr {
    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }
}

/// What a rank's device engine needs of the decomposition: the level
/// topologies and the restricted decimation rows between consecutive levels.
pub struct Topology {
    /// `levels[n]`: level `n` (level 0 the finest).
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
        let started = std::time::Instant::now();
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
        if std::env::var_os("STL_GPU_PROFILE").is_some() {
            eprintln!("  topology: blocks {:.1} s", started.elapsed().as_secs_f64());
        }
        let profile = std::env::var_os("STL_GPU_PROFILE").is_some();
        let t = std::time::Instant::now();
        let mut down = Vec::new();
        let mut up = Vec::new();
        for d in 0..levels.saturating_sub(1) {
            let (dn, upn) = decimation(geo, &lv[d], &lv[d + 1], threads);
            down.push(dn);
            up.push(upn);
        }
        if profile {
            eprintln!("  topology: decimation rows {:.1} s", t.elapsed().as_secs_f64());
        }
        Topology { levels: lv, down, up }
    }
}

/// Decimation rows from `fine` (level `d`) to `coarse` (level `d + 1`).
fn decimation(geo: &LocalGeometry, fine: &LevelTopo, coarse: &LevelTopo, threads: usize) -> (HostCsr, HostCsr) {
    let nside = fine.nside;
    let params = geo.down_params;
    let (fs, fe) = fine.owned;
    // Rows of the owned coarse cells (forward) and of every coarse cell whose
    // candidates may contain an owned fine cell (transpose: 3 coarse rings
    // around their parents, a superset), each computed once.
    let mut cells: Vec<i64> = (coarse.owned.0..coarse.owned.1).collect();
    if fs < fe {
        let fine_owned: Vec<i64> = (fs..fe).collect();
        cells.extend(par_map(&parents(&fine_owned), threads, |p| dilate(p, nside / 2, 3)));
    }
    cells.sort_unstable();
    cells.dedup();
    let profile = std::env::var_os("STL_GPU_PROFILE").is_some();
    let t = std::time::Instant::now();
    let rows: Vec<(Vec<i64>, Vec<f64>)> =
        par_map(&cells, threads, |cs| cs.iter().map(|&c| decimation_row(c, nside, &params)).collect());
    if profile {
        eprintln!("    level {}: {} rows computed in {:.1} s", fine.nside, cells.len(), t.elapsed().as_secs_f64());
    }
    // Forward: owned coarse cells, columns located in parallel.
    let owned_c: Vec<i64> = (coarse.owned.0..coarse.owned.1).collect();
    let fwd: Vec<(Vec<u32>, &Vec<f64>)> = par_map(&owned_c, threads, |cs| {
        cs.iter()
            .map(|&c| {
                let (f, w) = &rows[cells.binary_search(&c).unwrap()];
                (f.iter().map(|&v| find(&fine.ids, v, "decimation")).collect(), w)
            })
            .collect()
    });
    let mut dn = HostCsr { indptr: vec![0], ..Default::default() };
    for (cols, w) in fwd {
        dn.cols.extend(cols);
        dn.vals.extend_from_slice(w);
        // u32 row pointers: ~30 nonzeros per coarse row, so a rank holds
        // up to ~1.4e8 coarse cells of a level (nside 8192 needs >= 2 ranks).
        assert!(dn.cols.len() <= u32::MAX as usize, "decimation rows beyond u32 row pointers: use more ranks");
        dn.indptr.push(dn.cols.len() as u32);
    }
    // Transpose on the owned fine cells.
    let mut up = HostCsr { indptr: vec![0], ..Default::default() };
    if fs < fe {
        // Counting sort by owned fine cell; the coarse cells come in
        // increasing order, so they stay so within a row.
        let n = (fe - fs) as usize;
        let mut count = vec![0u32; n + 1];
        for (f, _) in &rows {
            for &fi in f {
                if fi >= fs && fi < fe {
                    count[(fi - fs) as usize + 1] += 1;
                }
            }
        }
        let mut sum = 0u64;
        for i in 0..n {
            sum += count[i + 1] as u64;
            assert!(sum <= u32::MAX as u64, "transposed decimation rows beyond u32 row pointers: use more ranks");
            count[i + 1] += count[i];
        }
        let total = count[n] as usize;
        let mut next = count.clone();
        up.cols = vec![0u32; total];
        up.vals = vec![0f64; total];
        for (&c, (f, w)) in cells.iter().zip(&rows) {
            let mut col = None;
            for (&fi, &wi) in f.iter().zip(w) {
                if fi >= fs && fi < fe {
                    let slot = &mut next[(fi - fs) as usize];
                    up.cols[*slot as usize] = *col.get_or_insert_with(|| find(&coarse.ids, c, "decimation transpose"));
                    up.vals[*slot as usize] = wi;
                    *slot += 1;
                }
            }
        }
        up.indptr = count;
    }
    (dn, up)
}
