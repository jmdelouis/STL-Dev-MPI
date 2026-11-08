//! Opt-in phase timers (`STL_PROFILE=1`): wall-clock seconds and counts
//! per phase name, accumulated in a process-wide table. A mark calls the
//! caller's synchronisation first (a GPU stream wait), so that asynchronous
//! work is charged to the phase that launched it. Off by default, when a
//! mark costs one atomic load.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static STATE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 off, 2 on
static TABLE: Mutex<BTreeMap<String, (f64, u64)>> = Mutex::new(BTreeMap::new());

/// Whether `STL_PROFILE` is set to a value other than `0` (read once).
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("STL_PROFILE").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        s => s == 2,
    }
}

/// A timer start (`None` when profiling is off).
pub fn start(sync: impl FnOnce()) -> Option<Instant> {
    enabled().then(|| {
        sync();
        Instant::now()
    })
}

/// Charges the time since `t0` to `name` (after `sync`) and restarts `t0`.
pub fn mark(t0: &mut Option<Instant>, name: &str, sync: impl FnOnce()) {
    if let Some(t) = t0 {
        sync();
        let now = Instant::now();
        let mut tab = TABLE.lock().unwrap();
        let e = tab.entry(name.to_string()).or_insert((0.0, 0));
        e.0 += (now - *t).as_secs_f64();
        e.1 += 1;
        *t = now;
    }
}

/// The table `{name: (seconds, count)}`, emptied.
pub fn take() -> BTreeMap<String, (f64, u64)> {
    std::mem::take(&mut *TABLE.lock().unwrap())
}
