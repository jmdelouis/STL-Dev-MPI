//! Fixture access for the tests: `fixtures/<suite>/<case>/<array>.npy`
//! written by `tools/dump_reference.py` (override the root with `STL_FIXTURES`).

#![allow(dead_code)]

use std::path::PathBuf;

pub fn root() -> PathBuf {
    match std::env::var_os("STL_FIXTURES") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
    }
}

pub struct Case {
    pub dir: PathBuf,
    pub meta: serde_json::Value,
}

impl Case {
    pub fn name(&self) -> &str {
        self.meta["name"].as_str().unwrap()
    }

    pub fn int(&self, key: &str) -> i64 {
        self.meta[key].as_i64().unwrap_or_else(|| panic!("{}: no integer {key}", self.name()))
    }

    pub fn str(&self, key: &str) -> &str {
        self.meta[key].as_str().unwrap()
    }

    pub fn has(&self, array: &str) -> bool {
        self.dir.join(format!("{array}.npy")).exists()
    }

    fn open(&self, array: &str) -> npyz::NpyFile<std::io::BufReader<std::fs::File>> {
        let path = self.dir.join(format!("{array}.npy"));
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        npyz::NpyFile::new(std::io::BufReader::new(file)).unwrap()
    }

    pub fn shape(&self, array: &str) -> Vec<usize> {
        self.open(array).shape().iter().map(|&s| s as usize).collect()
    }

    pub fn f64(&self, array: &str) -> Vec<f64> {
        self.open(array).into_vec::<f64>().unwrap()
    }

    pub fn i64(&self, array: &str) -> Vec<i64> {
        self.open(array).into_vec::<i64>().unwrap()
    }

    pub fn bool(&self, array: &str) -> Vec<bool> {
        self.open(array).into_vec::<u8>().unwrap().into_iter().map(|v| v != 0).collect()
    }
}

/// Cases of `suite` whose kind is `kind`.
pub fn cases(suite: &str, kind: &str) -> Vec<Case> {
    let dir = root().join(suite);
    let index = std::fs::read_to_string(dir.join("index.json")).unwrap_or_else(|e| {
        panic!("{}: {e} (run tools/dump_reference.sh {suite})", dir.display())
    });
    let index: serde_json::Value = serde_json::from_str(&index).unwrap();
    let out: Vec<Case> = index
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["kind"] == kind)
        .map(|m| Case { dir: dir.join(m["name"].as_str().unwrap()), meta: m.clone() })
        .collect();
    assert!(!out.is_empty(), "no {kind} case in {suite}");
    out
}

/// Rounding noise of a convolution output at pixel `id`: the direction's
/// colatitude goes through `acos(z)`, whose ulp error (glibc here, SLEEF in
/// torch) is amplified by `1 / sin(theta)`, and the bilinear weights are
/// linear in it over one pixel (`~nside`); the output sums `P` taps of O(1)
/// data and kernel.  The worst geometry fixture (nside 1024, equatorial
/// corner of face 4, 1.7e-11) is 0.45 of it; a porting error shows up as
/// orders of magnitude.
pub fn conv_noise(id: i64, nside: i64) -> f64 {
    let (theta, _) = stl_core::healpix::pix2ang_nest(id, nside);
    let s = theta.sin().max(1.0 / nside as f64);
    1e-12f64.max(f64::EPSILON * 128.0 * nside as f64 / s)
}

/// Largest absolute difference.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}
