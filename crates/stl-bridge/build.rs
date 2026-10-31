//! Records the git revision of the sources in the library
//! (`STL_BRIDGE_REVISION`, reported by `stlp_info`; "unknown" outside a git
//! checkout unless the variable is set at build time).

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=STL_BRIDGE_REVISION");
    let revision = std::env::var("STL_BRIDGE_REVISION").ok().or_else(|| {
        let head = git(&["rev-parse", "HEAD"])?;
        for f in ["HEAD", "index"] {
            if let Some(p) = git(&["rev-parse", "--path-format=absolute", "--git-path", f]) {
                println!("cargo:rerun-if-changed={p}");
            }
        }
        let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
        Some(if dirty { format!("{head}-dirty") } else { head })
    });
    println!("cargo:rustc-env=STL_BRIDGE_REVISION={}", revision.unwrap_or_else(|| "unknown".into()));
}
