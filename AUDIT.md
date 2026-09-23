# Allocation-collapse and intervention audit

The CPU audit checks twelve saved endpoints, six training schedules and 144 intervention records. It recomputes coordinate range, topic occupancy, Jensen-Shannon divergence and perplexity from included arrays. Exact seed-level intervals, cutoff sensitivity and removal of one seed at a time remain available. The analysis and figure build run without model checkpoint files or model training.

Setty at concentration 0.1 has strict collapse in two of three paired seeds; zero of three in other matched conditions does not rule out collapse. Seed trials are computational units on fixed data. Token and batch-context invariance are implementation checks, not causal evidence that attention itself produces collapse more often than a matched MLP.

The model and historical computation scripts are retained for inspection. Raw-data preprocessing, full historical training reproduction and recalculation of every graph/UMAP measurement are not part of the verified CPU workflow.

The source Setty and Dentate expression snapshots are not distributed, because exact upstream-version and annotation-transformation provenance has not been established for redistribution. This does not affect the saved-result CPU checks above. Model-state replay and historical training additionally require lawfully obtained, hash-matching snapshots; checkpoints alone are insufficient. The historical input digests are retained as provenance, not a claim of public availability or permission. See `SOURCE_DATA.md`.

## Verification boundary

The README specifies the commands and tested dependencies. `release-manifest.json` identifies the distributed files; historical producer hashes identify their original computations. All manuscript pages were rendered and programmatically checked for layout candidates. This is not a human visual sign-off. Scientific scope and known limitations are retained even when computational checks pass.
