//! Map I/O: NESTED `.npy` (float32 or float64) and HEALPix FITS binary
//! tables (the layout of `benchmark_synthesis.write_healpix_fits`; RING
//! maps are reordered to NESTED on reading).  The FITS code is pure Rust,
//! so that no cfitsio installation is needed on the cluster.

use std::io::{Read, Write};
use std::path::Path;
use stl_core::healpix::{order_of, ring2nest};

pub type Result<T> = std::result::Result<T, String>;

fn err<E: std::fmt::Display>(path: &Path) -> impl Fn(E) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// `nside` of a full-sky map of `npix` pixels.
pub fn nside_of(npix: usize) -> Result<i64> {
    let nside = ((npix / 12) as f64).sqrt().round() as i64;
    if 12 * nside * nside != npix as i64 || nside & (nside - 1) != 0 {
        return Err(format!("{npix} pixels is not a full-sky HEALPix map"));
    }
    Ok(nside)
}

/// A `.npy` array as f64 (any float or integer dtype), flattened.
pub fn read_npy(path: &Path) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).map_err(err(path))?;
    let npy = npyz::NpyFile::new(&bytes[..]).map_err(err(path))?;
    let dtype = npy.dtype();
    let npyz::DType::Plain(ts) = dtype else { return Err(format!("{}: structured arrays are not maps", path.display())) };
    let out: Vec<f64> = match (ts.type_char(), ts.size_field()) {
        (npyz::TypeChar::Float, 8) => npy.into_vec::<f64>().map_err(err(path))?,
        (npyz::TypeChar::Float, 4) => npy.into_vec::<f32>().map_err(err(path))?.into_iter().map(f64::from).collect(),
        (npyz::TypeChar::Int, 8) => npy.into_vec::<i64>().map_err(err(path))?.into_iter().map(|v| v as f64).collect(),
        (npyz::TypeChar::Int, 4) => npy.into_vec::<i32>().map_err(err(path))?.into_iter().map(f64::from).collect(),
        other => return Err(format!("{}: unsupported dtype {other:?}", path.display())),
    };
    Ok(out)
}

/// Write a flat little-endian float64 `.npy` (format 1.0).
pub fn write_npy(path: &Path, values: &[f64]) -> Result<()> {
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}", values.len());
    // Magic (6) + version (2) + length (2) + header + '\n', aligned on 64 bytes.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut buf = Vec::with_capacity(10 + header.len() + 8 * values.len());
    buf.extend(b"\x93NUMPY\x01\x00");
    buf.extend((header.len() as u16).to_le_bytes());
    buf.extend(header.bytes());
    for v in values {
        buf.extend(v.to_le_bytes());
    }
    std::fs::write(path, buf).map_err(err(path))
}

// ---------------------------------------------------------------------------
// FITS
// ---------------------------------------------------------------------------

const BLOCK: usize = 2880;

fn card(key: &str, value: &str) -> String {
    let c = if value.is_empty() { format!("{key:<8}") } else { format!("{key:<8}= {value:>20}") };
    format!("{c:<80}")
}

fn string_value(s: &str) -> String {
    format!("{:<20}", format!("'{s:<8}'"))
}

fn pad(buf: &mut Vec<u8>, fill: u8) {
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(fill);
    }
}

/// Write a full-sky NESTED map as a HEALPix binary table, one column of
/// float32 (`E`, as STL-Dev writes them) or float64 (`D`).
pub fn write_healpix_fits(path: &Path, values: &[f64], column: &str, double: bool) -> Result<()> {
    let nside = nside_of(values.len())?;
    let mut buf = Vec::new();
    for c in [card("SIMPLE", "T"), card("BITPIX", "8"), card("NAXIS", "0"), card("EXTEND", "T"), card("END", "")] {
        buf.extend(c.bytes());
    }
    pad(&mut buf, b' ');
    let width = if double { 8 } else { 4 };
    let header = [
        card("XTENSION", &string_value("BINTABLE")),
        card("BITPIX", "8"),
        card("NAXIS", "2"),
        card("NAXIS1", &width.to_string()),
        card("NAXIS2", &values.len().to_string()),
        card("PCOUNT", "0"),
        card("GCOUNT", "1"),
        card("TFIELDS", "1"),
        card("TTYPE1", &string_value(column)),
        card("TFORM1", &string_value(if double { "D" } else { "E" })),
        card("PIXTYPE", &string_value("HEALPIX")),
        card("ORDERING", &string_value("NESTED")),
        card("COORDSYS", &string_value("C")),
        card("EXTNAME", &string_value("xtension")),
        card("NSIDE", &nside.to_string()),
        card("FIRSTPIX", "0"),
        card("LASTPIX", &(values.len() - 1).to_string()),
        card("INDXSCHM", &string_value("IMPLICIT")),
        card("OBJECT", &string_value("FULLSKY")),
        card("END", ""),
    ];
    for c in header {
        buf.extend(c.bytes());
    }
    pad(&mut buf, b' ');
    for &v in values {
        if double {
            buf.extend(v.to_be_bytes());
        } else {
            buf.extend((v as f32).to_be_bytes());
        }
    }
    pad(&mut buf, 0);
    let mut f = std::fs::File::create(path).map_err(err(path))?;
    f.write_all(&buf).map_err(err(path))
}

/// Header cards of one HDU: `(keyword, raw value)`, and the header length.
fn parse_header(data: &[u8], start: usize) -> Option<(Vec<(String, String)>, usize)> {
    let mut cards = Vec::new();
    let mut at = start;
    loop {
        let c = data.get(at..at + 80)?;
        at += 80;
        let text = String::from_utf8_lossy(c);
        let key = text[..8].trim().to_string();
        if key == "END" {
            break;
        }
        if &text[8..10] == "= " {
            let raw = text[10..].trim();
            let value = if let Some(rest) = raw.strip_prefix('\'') {
                rest.split('\'').next().unwrap_or("").trim().to_string()
            } else {
                raw.split('/').next().unwrap_or("").trim().to_string()
            };
            cards.push((key, value));
        }
    }
    let len = (at - start).div_ceil(BLOCK) * BLOCK;
    Some((cards, len))
}

/// Read the first column of the first binary table of a HEALPix FITS file
/// as a NESTED map.
pub fn read_healpix_fits(path: &Path) -> Result<Vec<f64>> {
    let mut data = Vec::new();
    std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut data)).map_err(err(path))?;
    let bad = |m: &str| format!("{}: {m}", path.display());
    let mut at = 0;
    loop {
        let (cards, hlen) = parse_header(&data, at).ok_or_else(|| bad("no binary table"))?;
        let get = |k: &str| cards.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        let int = |k: &str| get(k).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let naxis = int("NAXIS");
        let size: usize = if naxis == 0 { 0 } else { (1..=naxis).map(|i| int(&format!("NAXIS{i}"))).product() };
        let data_len = (size + int("PCOUNT")).div_ceil(BLOCK) * BLOCK;
        if get("XTENSION").as_deref() == Some("BINTABLE") {
            let (row, nrows) = (int("NAXIS1"), int("NAXIS2"));
            let form = get("TFORM1").ok_or_else(|| bad("no TFORM1"))?;
            let (repeat, code) = form.split_at(form.find(|c: char| c.is_ascii_alphabetic()).ok_or_else(|| bad("bad TFORM1"))?);
            let repeat: usize = if repeat.is_empty() { 1 } else { repeat.parse().map_err(|_| bad("bad TFORM1"))? };
            let width = match &code[..1] {
                "E" => 4,
                "D" => 8,
                c => return Err(bad(&format!("column type {c} is not a float map"))),
            };
            let body = &data[at + hlen..];
            let mut values = Vec::with_capacity(nrows * repeat);
            for r in 0..nrows {
                for k in 0..repeat {
                    let o = r * row + k * width;
                    let b = body.get(o..o + width).ok_or_else(|| bad("truncated table"))?;
                    values.push(if width == 4 {
                        f32::from_be_bytes(b.try_into().unwrap()) as f64
                    } else {
                        f64::from_be_bytes(b.try_into().unwrap())
                    });
                }
            }
            let nside = nside_of(values.len())?;
            return match get("ORDERING").as_deref().unwrap_or("NESTED") {
                "NESTED" => Ok(values),
                "RING" => {
                    let order = order_of(nside);
                    let mut nest = vec![0.0; values.len()];
                    for (p, v) in values.iter().enumerate() {
                        nest[ring2nest(p as i64, order) as usize] = *v;
                    }
                    Ok(nest)
                }
                o => Err(bad(&format!("unknown ORDERING {o}"))),
            };
        }
        at += hlen + data_len;
    }
}

/// A map from `.npy` or `.fits` (by extension).
pub fn read_map(path: &Path) -> Result<Vec<f64>> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("fits") | Some("fit") => read_healpix_fits(path),
        _ => read_npy(path),
    }
}

// ---------------------------------------------------------------------------
// Slices: every rank reads and writes its own range of a map
// ---------------------------------------------------------------------------

fn is_fits(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("fits") | Some("fit"))
}

/// Element type of a `.npy` file and its byte order (big-endian when true).
#[derive(Clone, Copy, Debug)]
enum NpyType {
    F8(bool),
    F4(bool),
    I8(bool),
    I4(bool),
}

impl NpyType {
    fn size(self) -> usize {
        match self {
            NpyType::F8(_) | NpyType::I8(_) => 8,
            NpyType::F4(_) | NpyType::I4(_) => 4,
        }
    }

    fn value(self, b: &[u8]) -> f64 {
        macro_rules! get {
            ($t:ty, $be:expr) => {{
                let a = b.try_into().unwrap();
                if $be { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }
            }};
        }
        match self {
            NpyType::F8(be) => get!(f64, be),
            NpyType::F4(be) => get!(f32, be) as f64,
            NpyType::I8(be) => get!(i64, be) as f64,
            NpyType::I4(be) => get!(i32, be) as f64,
        }
    }
}

/// `(data offset, element type, number of elements)` of a `.npy` file.
fn npy_layout(path: &Path) -> Result<(u64, NpyType, usize)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(err(path))?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head).map_err(err(path))?;
    let bad = |m: &str| format!("{}: {m}", path.display());
    if &head[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file"));
    }
    let (hlen, start) = match head[6] {
        1 => (u16::from_le_bytes([head[8], head[9]]) as usize, 10),
        _ => (u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize, 12),
    };
    let mut text = vec![0u8; start + hlen];
    let mut f = std::fs::File::open(path).map_err(err(path))?;
    f.read_exact(&mut text).map_err(err(path))?;
    let dict = String::from_utf8_lossy(&text[start..]).to_string();
    let field = |key: &str| -> Option<String> {
        let at = dict.find(&format!("'{key}'"))? + key.len() + 2;
        let rest = dict[at..].trim_start().trim_start_matches(':').trim_start();
        Some(rest.to_string())
    };
    let descr = field("descr").ok_or_else(|| bad("no descr"))?;
    let d = descr.trim_start_matches('\'').split('\'').next().unwrap_or("");
    let (order, code) = d.split_at(d.len().min(1));
    let be = match order {
        ">" => true,
        "<" | "|" | "=" => false,
        _ => return Err(bad(&format!("unsupported dtype {d}"))),
    };
    let ty = match code {
        "f8" => NpyType::F8(be),
        "f4" => NpyType::F4(be),
        "i8" => NpyType::I8(be),
        "i4" => NpyType::I4(be),
        _ => return Err(bad(&format!("unsupported dtype {d}"))),
    };
    if field("fortran_order").is_some_and(|v| v.starts_with("True")) {
        return Err(bad("Fortran order"));
    }
    let shape = field("shape").ok_or_else(|| bad("no shape"))?;
    let inner = shape.trim_start_matches('(').split(')').next().unwrap_or("");
    let n: usize = inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>().map_err(|_| bad("bad shape")))
        .product::<Result<usize>>()?;
    Ok(((start + hlen) as u64, ty, n))
}

/// FITS binary table of a HEALPix map: `(data offset, row bytes, repeat,
/// element width, rows, nested)`.
fn fits_layout(path: &Path) -> Result<(u64, usize, usize, usize, usize, bool)> {
    let mut data = vec![0u8; 0];
    // Headers only: read until the first binary table header is complete.
    let mut f = std::fs::File::open(path).map_err(err(path))?;
    let bad = |m: &str| format!("{}: {m}", path.display());
    let mut at = 0;
    loop {
        while data.len() < at + 64 * BLOCK {
            let mut chunk = vec![0u8; BLOCK];
            match f.read(&mut chunk).map_err(err(path))? {
                0 => break,
                k => data.extend_from_slice(&chunk[..k]),
            }
        }
        let (cards, hlen) = parse_header(&data, at).ok_or_else(|| bad("no binary table"))?;
        let get = |k: &str| cards.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        let int = |k: &str| get(k).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let naxis = int("NAXIS");
        let size: usize = if naxis == 0 { 0 } else { (1..=naxis).map(|i| int(&format!("NAXIS{i}"))).product() };
        if get("XTENSION").as_deref() == Some("BINTABLE") {
            let form = get("TFORM1").ok_or_else(|| bad("no TFORM1"))?;
            let split = form.find(|c: char| c.is_ascii_alphabetic()).ok_or_else(|| bad("bad TFORM1"))?;
            let (repeat, code) = form.split_at(split);
            let repeat: usize = if repeat.is_empty() { 1 } else { repeat.parse().map_err(|_| bad("bad TFORM1"))? };
            let width = match &code[..1] {
                "E" => 4,
                "D" => 8,
                c => return Err(bad(&format!("column type {c} is not a float map"))),
            };
            let nested = get("ORDERING").as_deref().unwrap_or("NESTED") == "NESTED";
            return Ok(((at + hlen) as u64, int("NAXIS1"), repeat, width, int("NAXIS2"), nested));
        }
        at += hlen + (size + int("PCOUNT")).div_ceil(BLOCK) * BLOCK;
        // Skip the data of non-table HDUs by seeking.
        use std::io::{Seek, SeekFrom};
        data.clear();
        f.seek(SeekFrom::Start(at as u64)).map_err(err(path))?;
        data.resize(at, 0);
    }
}

/// Number of pixels of a map file without reading it.
pub fn map_len(path: &Path) -> Result<usize> {
    if is_fits(path) {
        let (_, _, repeat, _, rows, _) = fits_layout(path)?;
        Ok(repeat * rows)
    } else {
        Ok(npy_layout(path)?.2)
    }
}

fn read_at(path: &Path, offset: u64, len: usize) -> Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path).map_err(err(path))?;
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, offset).map_err(err(path))?;
    Ok(buf)
}

/// Pixels `[start, end)` of a NESTED map (`.npy` or HEALPix FITS), read
/// directly at their offset.  A RING FITS map is read whole and reordered
/// (no slice of RING pixels is a slice of NESTED pixels).
pub fn read_map_range(path: &Path, start: usize, end: usize) -> Result<Vec<f64>> {
    let n = end - start;
    if is_fits(path) {
        let (offset, row, repeat, width, _, nested) = fits_layout(path)?;
        if !nested {
            return Ok(read_healpix_fits(path)?[start..end].to_vec());
        }
        let mut out = Vec::with_capacity(n);
        // Rows of `repeat` pixels; read the covering rows.
        let (r0, r1) = (start / repeat, end.div_ceil(repeat));
        let bytes = read_at(path, offset + (r0 * row) as u64, (r1 - r0) * row)?;
        for p in start..end {
            let o = (p / repeat - r0) * row + (p % repeat) * width;
            let b = &bytes[o..o + width];
            out.push(if width == 4 { f32::from_be_bytes(b.try_into().unwrap()) as f64 } else { f64::from_be_bytes(b.try_into().unwrap()) });
        }
        return Ok(out);
    }
    let (offset, ty, total) = npy_layout(path)?;
    if end > total {
        return Err(format!("{}: range {start}..{end} beyond {total} elements", path.display()));
    }
    let size = ty.size();
    let bytes = read_at(path, offset + (start * size) as u64, n * size)?;
    Ok(bytes.chunks_exact(size).map(|b| ty.value(b)).collect())
}

/// Header of a flat float64 `.npy` of `n` elements.
fn npy_header(n: usize) -> Vec<u8> {
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({n},), }}");
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    let mut buf = Vec::with_capacity(10 + header.len());
    buf.extend(b"\x93NUMPY\x01\x00");
    buf.extend((header.len() as u16).to_le_bytes());
    buf.extend(header.bytes());
    buf
}

/// A map file written by slices: [`create`](Self::create) by one rank,
/// then [`write`](Self::write) of every rank's range, at its offset.
pub struct SlicedMap {
    pub path: std::path::PathBuf,
    offset: u64,
    fits: Option<bool>,
}

impl SlicedMap {
    /// The layout of the file (no I/O): `.npy` float64, or HEALPix FITS
    /// (`E` float32, or `D` when `double`) for a `.fits` path.
    pub fn new(path: &Path, double: bool) -> Self {
        let fits = is_fits(path).then_some(double);
        let offset = if fits.is_some() { 2 * BLOCK as u64 } else { npy_header(0).len() as u64 };
        SlicedMap { path: path.to_path_buf(), offset, fits }
    }

    /// Write the headers and size the file for `n` pixels.
    pub fn create(&self, n: usize, column: &str) -> Result<()> {
        let path = &self.path;
        let mut buf = match self.fits {
            None => npy_header(n),
            Some(double) => {
                let nside = nside_of(n)?;
                let mut b = Vec::new();
                for c in [card("SIMPLE", "T"), card("BITPIX", "8"), card("NAXIS", "0"), card("EXTEND", "T"), card("END", "")] {
                    b.extend(c.bytes());
                }
                pad(&mut b, b' ');
                let width = if double { 8 } else { 4 };
                for c in [
                    card("XTENSION", &string_value("BINTABLE")),
                    card("BITPIX", "8"),
                    card("NAXIS", "2"),
                    card("NAXIS1", &width.to_string()),
                    card("NAXIS2", &n.to_string()),
                    card("PCOUNT", "0"),
                    card("GCOUNT", "1"),
                    card("TFIELDS", "1"),
                    card("TTYPE1", &string_value(column)),
                    card("TFORM1", &string_value(if double { "D" } else { "E" })),
                    card("PIXTYPE", &string_value("HEALPIX")),
                    card("ORDERING", &string_value("NESTED")),
                    card("COORDSYS", &string_value("C")),
                    card("EXTNAME", &string_value("xtension")),
                    card("NSIDE", &nside.to_string()),
                    card("FIRSTPIX", "0"),
                    card("LASTPIX", &(n - 1).to_string()),
                    card("INDXSCHM", &string_value("IMPLICIT")),
                    card("OBJECT", &string_value("FULLSKY")),
                    card("END", ""),
                ] {
                    b.extend(c.bytes());
                }
                pad(&mut b, b' ');
                b
            }
        };
        assert_eq!(buf.len() as u64, self.offset, "header size");
        let data = match self.fits {
            None => 8 * n,
            Some(double) => (if double { 8 } else { 4 } * n).div_ceil(BLOCK) * BLOCK,
        };
        let f = std::fs::File::create(path).map_err(err(path))?;
        use std::io::Write;
        (&f).write_all(&std::mem::take(&mut buf)).map_err(err(path))?;
        f.set_len(self.offset + data as u64).map_err(err(path))
    }

    /// Write pixels `[start, start + values.len())`.
    pub fn write(&self, start: usize, values: &[f64]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let path = &self.path;
        let f = std::fs::OpenOptions::new().write(true).open(path).map_err(err(path))?;
        let (bytes, width): (Vec<u8>, usize) = match self.fits {
            None => (values.iter().flat_map(|v| v.to_le_bytes()).collect(), 8),
            Some(true) => (values.iter().flat_map(|v| v.to_be_bytes()).collect(), 8),
            Some(false) => (values.iter().flat_map(|&v| (v as f32).to_be_bytes()).collect(), 4),
        };
        f.write_all_at(&bytes, self.offset + (start * width) as u64).map_err(err(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_and_npy_roundtrip() {
        let dir = std::env::temp_dir().join(format!("stl-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let values: Vec<f64> = (0..12 * 16).map(|i| (i as f64).sin()).collect();
        let p = dir.join("m.fits");
        write_healpix_fits(&p, &values, "VALUE", true).unwrap();
        assert_eq!(read_map(&p).unwrap(), values);
        write_healpix_fits(&p, &values, "VALUE", false).unwrap();
        let single: Vec<f64> = values.iter().map(|&v| v as f32 as f64).collect();
        assert_eq!(read_map(&p).unwrap(), single);
        assert_eq!(std::fs::metadata(&p).unwrap().len() % 2880, 0);
        let q = dir.join("m.npy");
        write_npy(&q, &values).unwrap();
        assert_eq!(read_map(&q).unwrap(), values);
        // A big-endian float32 .npy (as numpy writes '>f4').
        let be = dir.join("be.npy");
        let mut raw = b"\x93NUMPY\x01\x00".to_vec();
        let mut header = format!("{{'descr': '>f4', 'fortran_order': False, 'shape': ({},), }}", values.len());
        while !(10 + header.len() + 1).is_multiple_of(64) {
            header.push(' ');
        }
        header.push('\n');
        raw.extend((header.len() as u16).to_le_bytes());
        raw.extend(header.bytes());
        raw.extend(values.iter().flat_map(|&v| (v as f32).to_be_bytes()));
        std::fs::write(&be, raw).unwrap();
        let single: Vec<f64> = values.iter().map(|&v| v as f32 as f64).collect();
        assert_eq!(map_len(&be).unwrap(), values.len());
        assert_eq!(read_map_range(&be, 5, 77).unwrap(), single[5..77].to_vec());
        // Slices: lengths, ranges, and files written by pieces.
        for path in [&p, &q] {
            let full = read_map(path).unwrap();
            assert_eq!(map_len(path).unwrap(), full.len());
            assert_eq!(read_map_range(path, 37, 101).unwrap(), full[37..101].to_vec());
        }
        for (name, double) in [("s.npy", false), ("s.fits", false), ("d.fits", true)] {
            let path = dir.join(name);
            let out = SlicedMap::new(&path, double);
            out.create(values.len(), "VALUE").unwrap();
            for (k, part) in values.chunks(50).enumerate() {
                out.write(50 * k, part).unwrap();
            }
            let expect: Vec<f64> =
                if name == "s.fits" { values.iter().map(|&v| v as f32 as f64).collect() } else { values.clone() };
            assert_eq!(read_map(&path).unwrap(), expect, "{name}");
            assert_eq!(read_map_range(&path, 60, 130).unwrap(), expect[60..130].to_vec(), "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
