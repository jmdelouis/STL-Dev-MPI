//! Minkowski functionals per pyramid level with the estimator of `stl-synth
//! --minkowski-weight` (see `stl_cli::minkowski`), on whole maps in one
//! process.  The thresholds are those of the reference map REF (the target).
//!
//!     stl-minkowski REF MAP... --sigma-out sigma.json      std over REF and the MAPs
//!     stl-minkowski REF MAP... --sigma sigma.json          z of the MAPs against REF
//!
//! Options: `--nside A-B` levels (default 64 to nside / 2), `--h X` and
//! `--eps X` (defaults 0.2 and 0.01; with `--sigma`, those of the file),
//! `--json OUT` every value.  With `--sigma`, prints per level and
//! functional the median and maximum of |z| over maps and thresholds and the
//! rms over the thresholds of the maps' median z, as `mf_levels.py`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use stl_cli::minkowski::{default_nu, Minkowski, SigmaFile, DEFAULT_EPS, DEFAULT_H};
use stl_mpi::io::{nside_of, read_map};

fn main() {
    if let Err(e) = run() {
        eprintln!("stl-minkowski: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut maps, mut range, mut h, mut eps) = (Vec::<PathBuf>::new(), None, DEFAULT_H, DEFAULT_EPS);
    let (mut sigma_out, mut sigma_in, mut json) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let mut it = argv.iter();
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--nside" => {
                let v = val()?;
                let (a, b) = v.split_once('-').unwrap_or((v, v));
                let p = |t: &str| t.parse::<usize>().map_err(|_| format!("--nside: '{t}'"));
                range = Some((p(a)?, p(b)?));
            }
            "--h" => h = val()?.parse().map_err(|_| "--h: not a number")?,
            "--eps" => eps = val()?.parse().map_err(|_| "--eps: not a number")?,
            "--sigma-out" => sigma_out = Some(val()?.into()),
            "--sigma" => sigma_in = Some(val()?.into()),
            "--json" => json = Some(val()?.into()),
            o if o.starts_with("--") => return Err(format!("unknown option {o}")),
            path => maps.push(path.into()),
        }
    }
    if maps.len() < 2 {
        return Err("usage: stl-minkowski REF MAP... [--nside A-B] [--h X] [--eps X] [--sigma-out F] [--sigma F] [--json F]".into());
    }
    let reference = read_map(&maps[0])?;
    let npix = reference.len();
    let nside = nside_of(npix)? as usize;
    let levels: Vec<usize> = (0..=(nside.trailing_zeros() as usize).saturating_sub(2))
        .filter(|&n| match range {
            Some((a, b)) => (a.min(b)..=a.max(b)).contains(&(nside >> n)),
            None => n >= 1 && (64..=nside / 2).contains(&(nside >> n)),
        })
        .collect();
    if levels.is_empty() {
        return Err("no level in the --nside range".into());
    }
    let level_nside: Vec<usize> = levels.iter().map(|n| nside >> n).collect();
    // With --sigma, the file's estimator; else a unit sigma (unused) with --h and --eps.
    let file = match &sigma_in {
        Some(p) => SigmaFile::read(p)?,
        None => {
            let nu = default_nu();
            let sigma = level_nside.iter().map(|&ns| (ns, vec![1.0; 3 * nu.len()])).collect();
            SigmaFile { nu, h, eps, sigma }
        }
    };
    let nt = file.nu.len();
    let label = sigma_in.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
    // [map][level, V0..V2, threshold], the reference first.
    let values: Vec<Vec<f64>> = stl_mpi::threads::run_ranks(1, |comm| -> Result<Vec<Vec<f64>>, String> {
        let mk = Minkowski::new(1.0, &reference, 0, npix, 1, &levels, Some((&file, label.as_str())), comm)?;
        let mut out = vec![mk.target_values()];
        for p in &maps[1..] {
            let m = read_map(p)?;
            if m.len() != npix {
                return Err(format!("{}: not the nside of {}", p.display(), maps[0].display()));
            }
            out.push(mk.functionals(&m, npix, comm).remove(0));
            eprintln!("{}", p.display());
        }
        Ok(out)
    })
    .remove(0)?;
    let per = 3 * nt;
    let split = |v: &[f64], l: usize| -> Vec<Vec<f64>> { (0..3).map(|k| v[l * per + k * nt..l * per + (k + 1) * nt].to_vec()).collect() };
    let mut report = serde_json::json!({
        "nu": file.nu, "h": file.h, "eps": file.eps,
        "reference": maps[0], "maps": maps[1..],
    });
    if let Some(out) = &sigma_out {
        // Standard deviation (ddof 1) over the reference and the maps.
        let n = values.len() as f64;
        let mut lv = serde_json::Map::new();
        for (l, &ns) in level_nside.iter().enumerate() {
            let mean: Vec<f64> = (0..per).map(|k| values.iter().map(|v| v[l * per + k]).sum::<f64>() / n).collect();
            let sd: Vec<f64> = (0..per)
                .map(|k| (values.iter().map(|v| (v[l * per + k] - mean[k]).powi(2)).sum::<f64>() / (n - 1.0)).sqrt())
                .collect();
            lv.insert(
                ns.to_string(),
                serde_json::json!({"sigma": split(&sd, 0), "mean": split(&mean, 0), "reference": split(&values[0], l)}),
            );
        }
        let s = serde_json::json!({"nu": file.nu, "h": file.h, "eps": file.eps, "reference": maps[0], "maps": maps[1..], "levels": lv});
        std::fs::write(out, serde_json::to_string_pretty(&s).unwrap()).map_err(|e| format!("{}: {e}", out.display()))?;
        eprintln!("sigma over {} maps -> {}", values.len(), out.display());
    }
    if sigma_in.is_some() {
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let n = v.len();
            if n == 0 { f64::NAN } else if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
        };
        let mut summary = BTreeMap::new();
        println!("z = (V - V(REF)) / sigma; median |z|, max |z| over maps x thresholds; rms over nu of the maps' median z");
        for (l, &ns) in level_nside.iter().enumerate() {
            let sig = &file.sigma[&ns];
            for k in 0..3 {
                let row = &sig[k * nt..(k + 1) * nt];
                let mx = row.iter().cloned().filter(|v| v.is_finite()).fold(0.0, f64::max);
                let kept: Vec<usize> = (0..nt).filter(|&t| row[t].is_finite() && row[t] > 0.05 * mx).collect();
                let z = |m: usize, t: usize| (values[m][l * per + k * nt + t] - values[0][l * per + k * nt + t]) / row[t];
                let abs: Vec<f64> = (1..values.len()).flat_map(|m| kept.iter().map(move |&t| z(m, t).abs())).collect();
                let sys = (kept.iter().map(|&t| median((1..values.len()).map(|m| z(m, t)).collect()).powi(2)).sum::<f64>()
                    / kept.len().max(1) as f64)
                    .sqrt();
                let (med, max) = (median(abs.clone()), abs.iter().cloned().fold(0.0, f64::max));
                println!("  level {ns:5} V{k}: {med:5.2} {max:6.2} {sys:5.2}");
                summary.insert(format!("{ns}_V{k}"), serde_json::json!({"median_abs_z": med, "max_abs_z": max, "rms_of_median_z": sys}));
            }
        }
        report["summary"] = serde_json::json!(summary);
    }
    if let Some(out) = &json {
        let mut lv = serde_json::Map::new();
        for (l, &ns) in level_nside.iter().enumerate() {
            lv.insert(ns.to_string(), serde_json::json!(values.iter().map(|v| split(v, l)).collect::<Vec<_>>()));
        }
        report["values"] = serde_json::Value::Object(lv);
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    Ok(())
}
