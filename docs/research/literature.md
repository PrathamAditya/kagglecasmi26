# Literature & tools — only what changes our decisions

| resource | one-line idea | where it plugs in |
|---|---|---|
| Li et al. 2021, Nat. Methods — spectral entropy similarity | Entropy-weighted similarity beats dot-product cosine for library search | exp_003 scorer |
| Li & Fiehn 2023, Nat. Methods — Flash entropy search (`ms_entropy`) | Indexed search over millions of spectra in ms; identity + open/hybrid modes | exp_003 (removes the 256 cap), exp_005 analog search |
| Wang et al. 2016, Nat. Biotech. — GNPS / modified cosine | Match fragments shifted by the precursor Δmass → finds structural analogs | exp_005 neighbors, exp_009 |
| Dührkop et al. — SIRIUS / CSI:FingerID | Fragmentation trees for formula; fingerprint prediction + database ranking | exp_005 formula filter, exp_006 design |
| Goldman et al. 2023, Nat. Mach. Intell. — MIST | Transformer over formula-annotated peaks → fingerprint | exp_006 upgrade path |
| Bushuiev et al. 2025, Nat. Biotech. — DreaMS / GeMS | Self-supervised spectrum embeddings pretrained on MassIVE | exp_006 alt. encoder (frozen features + FP head) |
| Bushuiev et al. 2024, NeurIPS D&B — MassSpecGym | Standard retrieval/de novo benchmark; de novo exact-match ≈ 0 | ADR-001 evidence |
| matchms | Spectrum cleaning/similarity utilities | exp_003 cleaning |

Add a row whenever you read something that could change a decision. If it can't, don't add it.
