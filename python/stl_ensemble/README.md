# STL ensemble L-BFGS v2: dimensionless SC variance ratios

Version: `2026-10-29-v2-variance-ratio`.

This separately versioned update changes only the SC population-variance
regularization and its validation/provenance workflow. It does not change native
SC definitions, the mean-loss sigma processing, power/PDF/MF constraints, or
joint native L-BFGS/strong-Wolfe.

See `../RUN_COMMANDS.md` for the exact GH200 adoption, validation, migration,
benchmark and pilot commands. Do not rerun the old v1 setup or modify its cache.
The provided runner reuses your existing isolated environment and exact GPU
bridge. It applies the explicit CPU-only NUMA policy required by your container.

Key new modules: `stl_port/sc_variance.py`, `stl_port/sc_reports.py`,
`stl_port/migrate_cache.py`; entry points `migrate_v1_cache.py` and
`validate_sc_ratio_native.py`.

See `V2_CHANGELOG.md`, `VALIDATION_REPORT.md` and `V2_SCOPE_AUDIT.json` for scope
and test evidence. Original v1 documentation/test evidence is retained under
`provenance/`; it is not evidence of v2 GPU acceptance.
