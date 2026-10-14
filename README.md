# ECB-HD-CR reproducibility package

Version 1.0.0, prepared 14 October 2026. This standalone package supports
**Two-timescale closed-loop energy-credit scheduling and routing for
energy-harvesting wireless sensor networks** by Zhongjun Ma, Jiayue Sun,
Pengfei Feng and Kangkang Zou. ECB is the Energy Credit Bank; HD denotes the
HHO–D3QN integration and CR denotes clustering and routing.

The package contains the implemented controller, five baselines, four
ablations, exact experiment specifications, input irradiance, individual run
records, source CSVs and analysis scripts. Manuscript files, referee
correspondence, tuning/search scripts and unrelated result versions are excluded.
The original diagram is supplied in its original layout as editable SVG/PDF.
The code, recorded results and reproduction instructions are publicly available at
[mayanzu/ECB-HD-CR](https://github.com/mayanzu/ECB-HD-CR). No Zenodo DOI has been assigned.

## Install

Run all commands from this directory. Python 3.12.10 and CPU execution were
used for validation; versions are recorded in `environment.json`.

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-figures-lock.txt
```

The lock file adds the official PyTorch CPU wheel index. For a compatible
unpinned environment use `requirements-figures.txt`; for analysis without
figure export use `requirements-lock.txt` or `requirements.txt`.
SVG-to-PDF export uses CairoSVG and needs the native Cairo library when not
available on the OS. No LaTeX, browser, API key or online data download is
needed to reproduce the numerical results. Fonts may be substituted on other
systems; supplied reference figures preserve the author's existing exports.
Numerical equivalence across different devices/library versions is not assumed.

## Reproduce from supplied results

```sh
python verify_package.py
python reproduce.py --workers 4
```

The second command restores 1,250 compressed raw records, runs 18 invariant
tests, recomputes legacy CSVs/tables and new overhead/bound statistics,
compares all 28 supplied CSV files numerically, and exports additional
quantitative figures as editable SVG followed by PDF. Outputs go to
`artifacts/`; distributed source CSVs and figure assets are left intact.

For numerical reproduction without CairoSVG figure export:

```sh
python reproduce.py --workers 4 --no-figures
```

Raw restoration verifies every record against the embedded checksum manifest
and does not replace differing existing experiments. Pickle files are Python
run records, not trained-weight checkpoints; load only trusted package files.
All metric units and the raw-record schema are documented in `DATA_DICTIONARY.md`.

## Rerun the simulation

The original matrix contains 950 runs across centered/stress/ablation/regime
shift/credit mismatch/NSRDB/gain sensitivity/scale phases. The deployment
audit contains 300 runs: five settings × six methods × ten seeds, 500 rounds.
`configs/experiment_matrix.json` lists all jobs and deduplicated full configs;
`experiments/experiment_specs.py` is the runner's parameter source.

For a quick end-to-end run (30 runs, one seed, 60 rounds):

```sh
python experiments/run_review_revision.py --workers 4 --smoke
```

Its outputs go to `results/review_revision_smoke/` and are excluded from
publication analysis. In a freshly extracted package, before restoring
supplied raw records, run the full experiments with:

```sh
python experiments/run_thesis_final.py --workers 4
python experiments/run_review_revision.py --workers 4 --seeds 10
```

Use a separate fresh extraction for simulation reruns when the supplied raw
results have already been restored. Both full runners write their versioned
`results/` directories; omitting `--resume` reruns matching filenames.
`--resume` can continue an interrupted run; the deployment runner checks
full config equality and the original runner checks version/run identity.
On GPU machines set `ECB_DEVICE=cpu` to match the tested device (PowerShell:
`$env:ECB_DEVICE='cpu'`; Linux/macOS: `export ECB_DEVICE=cpu`).

The supplied NSRDB input contains exactly the first 500 observations used by
the original NSRDB phase, with the upstream metadata and columns retained.
`data/provenance.json` records its source, coordinates, original/excerpt
checksums and transformation. Normalizing the CSV pathname in 120 raw config
snapshots removes author-specific local paths; numerical results are unchanged.
All other 1,130 raw records retain their original bytes. The raw ZIP manifest
records both source and distributed hashes.

## What the analysis measures

- Completion is delivered/offered tasks; FND is first-node-death round, capped
  at the horizon; energy/task is scheduling energy divided by delivered tasks.
- The original runs include data/forwarding energy and the modeled ECB
  256-bit uplink + 256-bit downlink cycle transaction where applicable.
- The added ideal centralized-BS protocol uses 128 status bits plus 32 bits
  per offered task, 64 route-command bits per nonempty batch (per task for
  ICR-HHODRL), and 64 topology-reception bits per clustered node every 50 rounds.
  It charges radio costs once and temporarily excludes nodes that cannot
  afford required round-level control transactions. It assumes reliable
  slotted links, known positions and ideal cycle-ledger access, and omits
  processor energy, contention, retransmissions and extra recovery signaling.
- The objective audit uses completion/lifetime/energy weights
  `(0.45, 0.35, 0.20)` and `E_ref = 1e-4 J`, independently of the training reward.
  `J_UB − J` bounds possible suboptimality under a hindsight relaxation;
  it is not an exact causal optimum or a joint convergence proof.
- New plots show mean ± sample SD over ten seeds. Paired t intervals describe
  the listed contrasts without a familywise significance claim. Replay-loss
  display smoothing uses 20 rounds; source CSV values are unsmoothed.
- Report seed offset is 720000. Data/topology/algorithm seeds and full config
  snapshots are retained for each run. HHO and deep Q learning remain stochastic.

## Files and evidence mapping

| Location | Contents / purpose |
|---|---|
| `algorithms/`, `core/`, `data/` | Proposed controller, baselines, ablations, radio/battery/task/harvest model |
| `experiments/` | Two paper experiment runners and their shared engine/specifications |
| `analysis/` | Objective relaxation, statistics export and figure/table generation |
| `tests/` | 18 implementation invariants |
| `configs/experiment_matrix.json` | Readable job/config inventory, 950 + 300 runs |
| `results/raw_runs.zip` | Compressed per-run evidence and 1,250-entry internal manifest |
| `results/thesis_final_v7/` | 23 supplied phase summary/statistics CSVs |
| `results/review_revision_v1/` | Five supplied CSVs and evaluation contract JSON |
| `assets/figures/` | Reference publication figure exports; three available SVG sources |
| `DATA_DICTIONARY.md`, `data/README.md` | Fields, units, input provenance and source-data mapping |
| `VALIDATION.json` | Validation performed on a freshly extracted copy |

Legacy analysis regenerates numerical tables, paired tests and a stress
tradeoff plot. It does not redraw the author's structural diagrams; those
are supplied as reference assets. SVG source files are supplied where
available, without claiming that all historical figures have SVG sources.

## Attribution and rights

Project software retains the existing MIT `LICENSE`. NSRDB input retains
its provider's terms and source attribution; see `THIRD_PARTY_NOTICES.md`.
Dependencies and fonts are not bundled. `CITATION.cff` supplies author/title
metadata without inventing publication or archive identifiers. Cite the
associated paper and add the repository-assigned identifier after deposition.
