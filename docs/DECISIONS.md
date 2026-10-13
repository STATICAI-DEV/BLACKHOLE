# BLACKHOLE Decisions

1. Use one private bucket, `BluStatic/BLACKHOLE`, with `DATA/` mounted at `/data` and `LAB/` mounted at `/lab`, matching the owner-approved `BLACKHOLE/DATA` layout while preserving storage separation.
2. Do not silently copy the legacy `BluStatic/BLACKHOLE-DATASETS` contents into the new mutable bucket; any migration requires separately verified manifests and hashes.
3. Keep the repository/model-neutral lab in `NO_MODEL` until the owner-supplied STATIC-0T asset is accessible and independently audited. No substitute architecture is invented.
4. Keep GitHub reporting manually dispatched until `HF_READ_TOKEN` is provisioned and a successful end-to-end run is verified. No unattended schedule is enabled.
5. Treat local synthetic fixtures as `TEST-ONLY`; never publish them as real training results.
