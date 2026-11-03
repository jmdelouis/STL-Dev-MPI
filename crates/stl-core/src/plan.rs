//! Coefficient plans: the list of ScatCov coefficients and their native
//! level (port of `PyramidScattering.ScatCoefficient` and `build_plan`).

/// Coefficient families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Pixel mean of the map.
    Mean,
    /// Pixel mean of the squared map.
    Var,
    /// First-order wavelet moments.
    S1,
    /// Second-order wavelet moments.
    S2,
    /// Covariances of a second wavelet of a modulus with a first wavelet
    /// (with the map under `has_fewer_convolutions`).
    S3,
    /// Covariances of two second wavelets of moduli.
    S4,
}

impl Family {
    /// Every family, in plan order.
    pub const ALL: [Family; 6] = [Family::Mean, Family::Var, Family::S1, Family::S2, Family::S3, Family::S4];

    /// Name of the family as in STL-Dev (`mean`, `var`, `S1` .. `S4`).
    pub fn name(self) -> &'static str {
        match self {
            Family::Mean => "mean",
            Family::Var => "var",
            Family::S1 => "S1",
            Family::S2 => "S2",
            Family::S3 => "S3",
            Family::S4 => "S4",
        }
    }

    /// Family of the STL-Dev name `name`, if any.
    pub fn parse(name: &str) -> Option<Self> {
        Family::ALL.into_iter().find(|f| f.name() == name)
    }
}

/// One coefficient: one channel pair and orientation tuple, all batches.
/// Unused scale/orientation indices are `None`, as in STL-Dev.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Coefficient {
    /// Coefficient family.
    pub family: Family,
    /// Level at which the coefficient is a pixel mean.
    pub native_dg: usize,
    /// First channel.
    pub c1: usize,
    /// Second channel.
    pub c2: usize,
    /// Scale `j1`.
    pub j1: Option<usize>,
    /// Scale `j2`.
    pub j2: Option<usize>,
    /// Scale `j3`.
    pub j3: Option<usize>,
    /// Orientation `l1`.
    pub l1: Option<usize>,
    /// Orientation `l2`.
    pub l2: Option<usize>,
    /// Orientation `l3`.
    pub l3: Option<usize>,
}

impl Coefficient {
    fn new(family: Family, native_dg: usize, c1: usize, c2: usize) -> Self {
        Coefficient { family, native_dg, c1, c2, j1: None, j2: None, j3: None, l1: None, l2: None, l3: None }
    }
}

/// Every coefficient of the operator (`J` scales, `L` orientations, scale
/// `j` at level `j_to_dg[j]`) for `channels` channels, cross pairs where
/// `cross[c1 * channels + c2]` (all by default), sorted by native level
/// (stable, so the order within a level is the enumeration order).
pub fn build_plan(
    j_count: usize,
    l_count: usize,
    j_to_dg: &[usize],
    channels: usize,
    cross: Option<&[bool]>,
    families: &[Family],
) -> Vec<Coefficient> {
    assert!(channels >= 1, "channels must be positive");
    assert_eq!(j_to_dg.len(), j_count);
    if let Some(c) = cross {
        assert_eq!(c.len(), channels * channels, "cross matrix has the wrong dimensions");
    }
    let has = |f: Family| families.contains(&f);
    let crossed = |a: usize, b: usize| cross.is_none_or(|c| c[a * channels + b]);
    let mut plan = Vec::new();
    for family in [Family::Mean, Family::Var] {
        if has(family) {
            plan.extend((0..channels).map(|c| Coefficient::new(family, 0, c, c)));
        }
    }
    for (j3, &dg) in j_to_dg.iter().enumerate() {
        for c1 in 0..channels {
            for c2 in c1..channels {
                if !crossed(c1, c2) {
                    continue;
                }
                for family in [Family::S1, Family::S2] {
                    if has(family) {
                        plan.extend((0..l_count).map(|l| Coefficient {
                            j1: Some(j3),
                            l1: Some(l),
                            ..Coefficient::new(family, dg, c1, c2)
                        }));
                    }
                }
                let pairs: Vec<(usize, usize)> = if c1 == c2 { vec![(c1, c2)] } else { vec![(c1, c2), (c2, c1)] };
                for (ca, cb) in pairs {
                    for j2 in 0..=j3 {
                        if has(Family::S3) {
                            for l2 in 0..l_count {
                                for l3 in 0..l_count {
                                    plan.push(Coefficient {
                                        j1: Some(j2),
                                        j3: Some(j3),
                                        l1: Some(l2),
                                        l3: Some(l3),
                                        ..Coefficient::new(Family::S3, dg, ca, cb)
                                    });
                                }
                            }
                        }
                        if has(Family::S4) {
                            for j1 in 0..=j2 {
                                for l1 in 0..l_count {
                                    for l2 in 0..l_count {
                                        for l3 in 0..l_count {
                                            plan.push(Coefficient {
                                                j1: Some(j1),
                                                j2: Some(j2),
                                                j3: Some(j3),
                                                l1: Some(l1),
                                                l2: Some(l2),
                                                l3: Some(l3),
                                                ..Coefficient::new(Family::S4, dg, ca, cb)
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    plan.sort_by_key(|c| c.native_dg);
    plan
}
