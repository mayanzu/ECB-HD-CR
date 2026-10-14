# Third-party notices

## National Solar Radiation Database input

Credit: U.S. Department of Energy and the National Solar Radiation Database
(NSRDB), supplied by NREL (now the National Laboratory of the Rockies).
The bundled input is an excerpt of the author's original NSRDB export;
coordinates, version, coverage and checksum lineage are in `data/provenance.json`.
The original NSRDB metadata header is retained. This project does not assert
ownership of the provider's input data or relicense it under MIT.
The complete current provider data-permission notice is retained in
`data/UPSTREAM_NOTICE.txt`; retain it with any copy of the input excerpt.

Provider documentation and source terms (checked 14 October 2026):

- https://nsrdb.nlr.gov/
- https://developer.nlr.gov/docs/solar/nsrdb/
- https://www.nlr.gov/comm-standards/web/data

Use provider data according to its applicable terms and attribution notices;
the software license in this repository covers the project's own software.

## Dependencies and typography

NumPy, PyTorch, SciPy, pandas, Matplotlib, PyYAML, tqdm and CairoSVG are
installed separately and retain their respective upstream licenses.
No third-party library binaries or font files are bundled. Supplied figure
exports retain existing typography; availability of Arial, Cambria and
Times New Roman on another machine is not guaranteed.
