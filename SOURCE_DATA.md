# External source-expression inputs

The default CPU analysis consumes distributed model allocations, schedules,
probe outputs and summary records. It does not need source-expression snapshots
or model checkpoints. The following historical expression files are not
distributed on GitHub or in the Zenodo archive:

- `setty.npz`: SHA-256 `1efd62dcfb9241358b4a43050406df001d6af76b7c88ee3dfc7980357c302496`.
- `dentate.npz`: SHA-256 `242b2104dad518a1950714dd1123d9dbdf0709461033547c7939e16597b9620f`.

They contain source-derived expression, cell/gene identifiers and annotations.
The exact upstream-version and annotation-transformation chain has not been
established for redistribution. Original input manifests retain these hashes
as provenance; `publication-manifest.json` lists them separately as external
inputs, not required distributed files. A checksum is not a license.

The manuscript cites the source studies. Upstream references include the
[Setty hematopoiesis project](https://explore.data.humancellatlas.org/projects/091cf39b-01bc-42e5-9437-f419a66c8a45),
the [versioned hematopoiesis dataset](https://doi.org/10.6084/m9.figshare.14465559.v1),
and [Hochgerner dentate gyrus accession GSE104323](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104323).
They are not asserted to regenerate the exact excluded snapshots. Their own
terms, not the author's CC BY 4.0 grant, govern reuse.

Historical `train.py` and `run_probes.py` accept `--input-dir DIRECTORY` for
separately and lawfully obtained snapshots, check both hashes and stop if inputs
are missing or different. They write to `reproduced/` by default rather than
overwriting evidence. The probe also needs final model states from this study's
checkpoint archive; the training implementation needs CUDA. Neither workflow
is included in the verified public CPU analysis or a claim of end-to-end
source-data preprocessing reproduction.

The two retained `TRAINING_PROTOCOL.md` copies are historical freeze records.
Their phrase "included fixed input files" refers to the historical experiment,
not this public release. Read their input references subject to this exclusion;
the protocol's previously documented publication normalization is unchanged,
and historical digests remain distinguished from current public-file digests.

Model-state probes default to the archived `evidence/reproduction/training/`
states. Use `--checkpoint-dir reproduced/training` explicitly for separately
generated training outputs. Selected state hashes are recorded alongside the
new probe outputs; selecting a new state does not certify historical equivalence.
