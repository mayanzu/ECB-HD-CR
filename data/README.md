# Irradiance and generated task inputs

`nsrdb_hefei.csv` is a 500-observation excerpt of the author's supplied NSRDB
CSV. It retains the original two metadata rows, column header and first 500
data rows. The location metadata are 31.81° N, 117.22° E, location ID 1096422,
Anhui, China, version 3.2.0. These observations start in January 2015; the
unused later observations of the full source file are not bundled.

The input is satellite/model-derived NSRDB irradiance, not a sensor experiment
collected by the paper's authors. Original source: https://nsrdb.nlr.gov/;
API documentation: https://developer.nlr.gov/docs/solar/nsrdb/.
`provenance.json` gives exact original and excerpt checksums and coverage.

`nsrdb_loader.py` locates GHI (W/m²), averages sub-hourly groups if needed,
then uses the next hourly GHI value for each simulation round. Each value is
integrated over 60 seconds and multiplied by the scenario scale; 500 rounds
are therefore a measured-shape replay rather than 500 physical hours of
harvested energy. This mapping is retained from the original experiment.

Other inputs are generated reproducibly by `solar_model.py` and `task_model.py`
using the recorded data seeds. Task sizes are bits; inter-node distances are
metres; radio/harvest/background energies are joules. Task arrivals share the
same environment stream among algorithms in a paired scenario/seed.

The project software MIT license does not replace upstream NSRDB data terms.
See `THIRD_PARTY_NOTICES.md` for attribution and provider links. The supplied
excerpt allows reproduction without API credentials or a network download.
The provider's full data notice is included in `UPSTREAM_NOTICE.txt`.
