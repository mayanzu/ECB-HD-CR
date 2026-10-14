# Evidence fields, units and figure mapping

Two run versions are kept separate. `thesis_final_v7` has 950 raw records;
`review_revision_v1` has 300. The ZIP manifest records per-file SHA-256 and
the source-file hash before optional pathname normalization. No runs were
excluded from these two reported matrices.

## Raw pickle records

| Field | Meaning / units |
|---|---|
| `method`, `N`, `run_id` | Algorithm label, node count, report-seed index |
| `data_seed`, `topology_seed`, `algorithm_seed` | Separate integer RNG seeds |
| `config_snapshot` | Complete simulator/scenario parameters; energies J, distances m, sizes bits |
| `round`, `alive_count` | Recorded round index and alive-node counts |
| `fnd`, `hnd`, `lnd` | First/half/last node death, rounds; horizon used when event is not observed |
| `total_tasks`, `completed_tasks`, `completion_rate` | Offered count, delivered count, ratio |
| `total_energy` | Total scheduling energy, J; includes applicable control components |
| `background_energy`, `total_physical_energy` | Background energy and scheduling + background energy, J |
| `control_energy` | Existing modeled ECB cycle-transaction radio energy, J |
| `budget_deviation` | Original recorded credit-budget deviation metric |
| `alive_node_rounds` | Sum of alive counts across recorded rounds |
| `pre_switch_*`, `post_switch_*`, `switch_round` | Regime-shift count/completion/survival fields when applicable |
| `deployment_energy`, `deployment_energy_components` | Added telemetry/commands/topology radio energy, J (new version) |
| `deployment_message_counts`, `control_failed_transactions` | Added ideal protocol message/transaction counters |
| `diagnostics` | Per-round proposed-controller observations: bill reference/mean bill J, credit, lambda, epsilon, replay loss and counts |

There are no trained network checkpoints in these records. An unused or
unobserved diagnostic is represented as NaN in exported CSVs, not zero.

## Supplied CSVs

| Location | Meaning |
|---|---|
| `thesis_final_v7/*/summary.csv` | Per-seed records for eight experiment phases; ablation/NSRDB synthetic sides reuse the same original runs |
| `thesis_final_v7/*/aggregated_stats.csv` | Per-method/scenario means and sample SD; sensitivity uses its original summary schema |
| `thesis_final_v7/*/relative_gains.csv` | Original relative improvements; not present for sensitivity |
| `review_revision_v1/deployment_per_seed.csv` | 300 per-seed records; includes scheduling energy J, physical energy J, energy/task µJ, protocol costs J and ratios |
| `review_revision_v1/deployment_summary.csv` | 30 method/setting cells × ten seeds, mean/sample SD |
| `review_revision_v1/deployment_diagnostics.csv` | 50 proposed-method runs × 500 rounds; original unsmoothed diagnostic values |
| `review_revision_v1/legacy_bound_audit.csv` | 110 original proposed-method objective values, relaxed upper bounds and differences |
| `review_revision_v1/paired_deployment_differences.csv` | Proposed minus D3QN paired completion/FND/objective and individual 95% t intervals; energy reduction compares means |

The bound column is a tracewise hindsight relaxation. It drops battery/credit/
cluster restrictions and downstream/control costs and optimistically sets
lifetime to the horizon. Its difference from the measured objective is an
upper bound on possible suboptimality, not a measured exact optimality gap.

## Mapping to figures and tables

- Centered/stress/ablation/regime/mismatch/NSRDB/sensitivity/scale numerical
  tables and paired tests: `analysis/generate_thesis_artifacts.py`, using
  raw-derived legacy exports. Files are generated under `artifacts/legacy/generated/`.
- Existing ablation/stress/credit-robustness publication figures:
  corresponding phase CSVs in `thesis_final_v7`; original PDF exports are
  supplied in `assets/figures/`. Historical export styling is preserved there.
- Deployment comparison figure/table: `deployment_per_seed.csv` and
  `deployment_summary.csv`; figure generator `analysis/draw_review_figures.py`.
- Controller diagnostic figure: `deployment_diagnostics.csv`; same generator.
- Objective upper-bound table: legacy bound audit and proposed-method rows
  of `deployment_per_seed.csv`.
- Framework/network/ECB mechanism diagrams: structural illustrations supplied
  as reference assets; the ECB mechanism retains the author's SVG source.

The standalone analysis writes LaTeX table fragments as text without requiring
a TeX compiler. Figure error bars/bands are sample SD over ten report seeds.
