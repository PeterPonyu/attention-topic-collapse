# Single-topic collapse in a within-cell attention topic model

## Repository, archive and citation

**Archival status:** the Zenodo DOI below is reserved, but the archival record
is not yet published because checkpoint-volume transfers remain incomplete.
The source and manuscript are publicly available in GitHub release `v1.0.1`;
Zenodo checkpoint downloads are not yet available. The release checksum files
describe the immutable `v1.0.1` tag and its source ZIP, not this subsequent
status note on the default branch.

Study repository: [PeterPonyu/attention-topic-collapse](https://github.com/PeterPonyu/attention-topic-collapse). Versioned archive: [10.5281/zenodo.22914481](https://doi.org/10.5281/zenodo.22914481). Use `CITATION.cff` for release 1.0.1; `release-manifest.json` records distributed-file checksums. The archive and source repository describe this study only.

Author-owned software is MIT licensed. The author's manuscript, figures, generated results and model weights are CC BY 4.0. Source-study data, labels, annotations and other third-party materials retain their original terms; they are not relicensed. Read `LICENSE` and `NOTICE.md` before reusing mixed-content files.

This research package examines when a within-cell attention encoder produces
nearly identical topic allocations across held-out cells, and what token and
batch-context interventions reveal about its implementation. The matched study
compares Dirichlet concentrations 0.1 and 1.0 on fixed Setty and Dentate data
objects, with three paired training seeds per condition. Historical sweeps on
four fixed backgrounds provide context at other concentrations and training
budgets. Their endpoints remain separate from the matched experiment.

The reproducible product is analysis of saved results: exact seed-level collapse
intervals, allocation and probe audits, manuscript figures, and the article
build. These operations run on a CPU, perform no training, and require neither
a GPU nor model checkpoints. Large `.pt` files are distributed separately in
this article's Zenodo checkpoint archive; the default GitHub analysis and figure
commands do not read them.

For operations that consume model states, download this article's checkpoint
archive from its Zenodo DOI record and restore the package-relative paths
listed under `optional_checkpoints` in `publication-manifest.json`. A GitHub
source checkout alone does not supply those model states.
Model-state probes and historical training also require separately and lawfully
obtained, hash-matching Setty and Dentate expression snapshots. Neither GitHub
nor the Zenodo archive distributes those snapshots. `SOURCE_DATA.md` records
their identities, upstream references and the unresolved exact-version
provenance. Downloading checkpoints alone is insufficient for those operations.

## CPU analysis

Use Python 3.11 or newer and the versions in `requirements-analysis.txt` (NumPy,
SciPy and scikit-learn). The recorded validation used Python 3.13.5. From this
package's root directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python evidence/collapse_ci.py
python scripts/robustness_audit.py
python scripts/verify_publication.py
```

`collapse_ci.py` uses only the Python standard library and
`evidence/training_results.json`. It validates the saved strict-collapse flags
and writes two-sided exact Clopper–Pearson 95% intervals to
`evidence/collapse_ci.json`, treating training seeds as the trials.

`robustness_audit.py` reads the 12 saved 450-by-10 allocation arrays, six saved
training schedules, and 24 probe-array archives. It recomputes coordinate range,
topic occupancy, pairwise Jensen–Shannon divergence and perplexity; checks all
144 intervention records; and reports threshold sensitivity, removal of one
seed at a time, and nearest-neighbour tie diagnostics. It creates `validation/`
if necessary and writes `validation/robustness-audit.json`. It reads no `.pt`
file and never imports the model or training code.

The identity check uses only the Python standard library. It verifies the
analysis-input and source-snapshot hashes in `publication-manifest.json`.
Checkpoint presence is optional; absent checkpoints are reported without
blocking analysis. Historical manifests and source digests are preserved as
records of the original computation.
The two external expression-input digests are reported separately, not counted
as distributed or verified files.

## Figures and article build

Figure regeneration requires R with `ggplot2`, `gridExtra`, `jsonlite`, the
standard `grid` package, and Cairo graphics support. The tested versions are
R 4.3.3, ggplot2 4.0.3, gridExtra 2.3 and jsonlite 2.0.0. Install missing R
packages in an existing R environment with:

```sh
Rscript -e 'install.packages(c("ggplot2", "gridExtra", "jsonlite"), repos="https://cloud.r-project.org")'
```

Install Arial and the system librsvg, Cairo and GObject shared libraries. The
SVG renderer uses Python's standard library to call those libraries; Python
PyGObject and Pycairo packages are not required. On Debian/Ubuntu the relevant
runtime library packages are `librsvg2-2`, `libcairo2` and `libglib2.0-0`
(the GLib package name can vary by distribution release). `python3` must be on
`PATH`; set `PYTHON` to another Python executable if needed.

```sh
Rscript scripts/build_figures.R
bash build.sh
```

The first command rebuilds seven data-figure PDF/PNG pairs and the vector
architecture PDF from package-local JSON, CSV and SVG inputs. It updates
`scripts/figure-data-counts.json`. Fixed UMAP coordinates are plotted as saved;
UMAP fitting and raw-data preprocessing are not repeated.

The build command regenerates the figures, runs XeLaTeX, BibTeX and two further
XeLaTeX passes, then copies `build_independent/paper.pdf` to `paper.pdf`. It needs
a TeX installation containing the packages listed in
`standalone-assets/preamble.tex`, including `fontspec`, `natbib`, `tikz`,
`siunitx`, `adjustbox` and `microtype`. Arial must also be visible to XeLaTeX.
The build overwrites generated figures and the article PDF, but does not fit a
model or regenerate the scientific evidence. Pre-rendered figures are included.

## Inputs and scientific limits

- `evidence/training_results.json`, `evidence/probe_results.json` and
  `evidence/reproduction/` contain the matched endpoints, loss traces,
  schedules, saved allocations and probe-output arrays. Source normalized/count
  expression snapshots and their cell/gene identifiers are not distributed.
- `scripts/inputs/` and `evidence/historical_rows.csv` contain the historical
  scores and fixed plotting coordinates. `figure-build.json` lists figure
  inputs. The stored historical scores are plotted without rescoring.
- `evidence/reproduction/source/` retains the exact model implementation.
  `train.py` and `run_probes.py` in that directory's parent are historical
  computation records, not default analysis commands. The training script
  requires CUDA and PyTorch; both scripts require non-distributed expression
  snapshots via `--input-dir DIRECTORY`. Replaying model probes additionally
  requires the separate final checkpoints. Both reject missing or mismatched
  expression inputs before computation; new outputs go under `reproduced/`,
  not over the archived evidence. Neither operation is part of the verified CPU
  analysis above. The model source is unchanged; these wrapper input/output
  safeguards are documented release adaptations, not historical model changes.
  The probe's `--checkpoint-dir DIRECTORY` selects a root containing each arm's
  `last.pt`; by default it selects the archived `evidence/reproduction/training/`.
  To inspect separately generated training outputs, explicitly select
  `--checkpoint-dir reproduced/training`. The probe records selected state
  hashes in its output; newly trained states are not automatically equivalent
  to the published states or validated by the saved-result CPU audit.
- `evidence/PROTOCOL.md` records the initial-model probe design.
  `evidence/TRAINING_PROTOCOL.md` specifies the subsequent matched training
  study and trained-state probes. `evidence/FUTURE_EXPERIMENT_FREEZE.md`
  describes an unexecuted MLP comparison and larger seed grid; it contains no
  new experimental results.

Strict collapse requires one occupied argmax topic and maximum across-cell
coordinate range at most `1e-8`. A pairwise divergence at most `1e-12` flags
degenerate allocation separately. In the matched study Setty at concentration
0.1 collapses in 2/3 seeds; the other three conditions have 0/3. Three seeds
give wide intervals, and 0/3 does not rule out collapse. Historical degeneracy
counts are not interchangeable with these matched strict-collapse counts.

The 450 held-out cells and training seeds are computational units within fixed
public data objects, not biological replicates. The preprocessing and split
are frozen historical inputs, not a claim of training-only feature selection.
Token/batch probes test implementation behavior and do not identify a cause of
collapse or establish that attention fails more often than a matched MLP.
This package does not claim end-to-end regeneration of every historical
training sweep, raw-data preprocessing, graph score or UMAP embedding.

## Optional checkpoint download and reconstruction

Release 1.0.1 changes only checkpoint distribution and documentation; scientific results, model-state bytes and the manuscript are unchanged from 1.0.0. To make large downloads recoverable, the study-specific Zenodo record stores the checkpoint ZIP as ordered 20 MiB volumes. `checkpoint-downloads.json` lists each volume URL, byte count and SHA-256, plus the reconstructed ZIP identity. These volumes belong only to this study and must not be mixed with other releases.

From the repository root, use the Python standard-library helper:

```bash
python3 scripts/restore_checkpoints.py --download-dir ../checkpoint-downloads --download
unzip -n reproduced/checkpoints.zip -d .
```

If the volumes were downloaded separately, omit `--download`. The helper validates every volume and the reconstructed ZIP and refuses to overwrite different existing files. It does not deserialize models or execute training. After extraction, the original package-relative `.pt` paths and hashes are unchanged. Default CPU saved-result analyses do not need these downloads.
