"""
Real NSRDB solar-trace loader.

This module turns a real NSRDB (NREL PSM3) CSV export into a per-round
harvested-energy base trace that is a *drop-in replacement* for
``SolarTraceGenerator.generate_synthetic_trace``.

Design goal: comparability.
Only the irradiance *source* changes (synthetic diurnal sine -> measured GHI).
The downstream pipeline in ``run_main_experiments.py`` is untouched:

    base_trace  = nsrdb_base_trace(csv, total_rounds)   # <-- this module
    spatial     = solar_gen.add_spatial_variation(base_trace, N)   # per-node phi_i
    spatial    *= energy_scale_factor                              # scenario scaling
    (optional regime shift)
    harvest_t   = spatial[:, t]

Round mapping
-------------
One round is mapped to the next hourly measured GHI sample, so a 500-round run
uses the first 500 observations (~21 days of weather ordering).  Each sampled
irradiance is nevertheless integrated over the simulator's 60 s round.  This
preserves the synthetic experiment's raw energy scale and should be described
as a measured-shape replay, not as 500 physical hours of harvested energy.

The irradiance->energy conversion deliberately reuses the same
``SolarHarvester`` constants and the same 60 s integration window as the
synthetic generator, so the raw Joule magnitude stays in the synthetic range
and every scenario's ``energy_scale_factor`` remains valid without retuning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.solar_model import SolarHarvester


def load_ghi_series(csv_path: str | Path) -> np.ndarray:
    """Load an hourly GHI series (W/m^2) from an NSRDB PSM3 CSV export.

    NSRDB PSM3 CSV layout:
        row 0 : metadata field names
        row 1 : metadata values
        row 2 : column header  (Year, Month, Day, Hour, Minute, GHI, DNI, ...)
        row 3+: data

    Sub-hourly (e.g. 30 min) exports are averaged down to hourly resolution.

    Returns
    -------
    np.ndarray
        1-D array of hourly GHI in W/m^2 (non-negative, NaNs dropped).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NSRDB CSV not found: {csv_path}\n"
            "Download an hourly GHI export from https://nsrdb.nrel.gov/data-viewer "
            "(PSM3, pick a location/year, CSV), or run with --make-demo-csv to "
            "generate a smoke-test file."
        )

    # The PSM3 export has two metadata rows before the real header.
    df = pd.read_csv(csv_path, skiprows=2)

    # Locate the GHI column case-insensitively (exact 'GHI', not 'Clearsky GHI').
    ghi_col = None
    for col in df.columns:
        if str(col).strip().lower() == "ghi":
            ghi_col = col
            break
    if ghi_col is None:
        raise ValueError(
            f"No 'GHI' column found in {csv_path}. Columns: {list(df.columns)}"
        )

    # Average sub-hourly samples to hourly if a Minute column shows >0 values.
    minute_col = next((c for c in df.columns if str(c).strip().lower() == "minute"), None)
    if minute_col is not None and (df[minute_col] != 0).any():
        keys = [c for c in df.columns
                if str(c).strip().lower() in ("year", "month", "day", "hour")]
        if keys:
            df = df.groupby(keys, sort=False)[ghi_col].mean().reset_index()

    ghi = pd.to_numeric(df[ghi_col], errors="coerce").to_numpy(dtype=float)
    ghi = ghi[~np.isnan(ghi)]
    ghi = np.clip(ghi, 0.0, None)
    if ghi.size == 0:
        raise ValueError(f"GHI column in {csv_path} contained no usable values.")
    return ghi


def nsrdb_base_trace(
    csv_path: str | Path,
    total_rounds: int,
    seconds_per_round: float = 60.0,
    panel_area: float = 0.01,
    efficiency: float = 0.20,
) -> np.ndarray:
    """Build a per-round harvested-energy base trace (Joules) from real GHI.

    One round consumes the next hourly GHI observation. If ``total_rounds``
    exceeds the available observations, the measured pattern is tiled. The GHI
    value is converted with ``seconds_per_round`` (60 s by default), rather
    than being integrated over a full hour.

    The raw Joule scale matches ``generate_synthetic_trace`` so each scenario's
    ``energy_scale_factor`` applies unchanged.
    """
    ghi = load_ghi_series(csv_path)

    # Map round index -> hourly GHI sample (wrap around for long horizons).
    idx = np.arange(total_rounds) % ghi.size
    ghi_per_round = ghi[idx]

    harvester = SolarHarvester(panel_area=panel_area, efficiency=efficiency)
    power = ghi_per_round * harvester.panel_area * harvester.efficiency  # W
    energy = power * seconds_per_round  # J per round
    return energy.astype(float)


def synthesize_demo_csv(out_path: str | Path, days: int = 30, seed: int = 0) -> Path:
    """Write a PSM3-shaped CSV with *fake but realistic* hourly GHI.

    For SMOKE-TESTING the pipeline only -- this is NOT real data and must not
    be used for any reported result. It lets the loader/runner be validated
    end-to-end before a real NSRDB export is downloaded.
    """
    out_path = Path(out_path)
    rng = np.random.RandomState(seed)

    rows = []
    for d in range(days):
        # Random daily cloud factor to create multi-day non-stationarity.
        cloud = rng.uniform(0.45, 1.0)
        for h in range(24):
            # Clear-sky-ish diurnal bell peaking at ~1 kW/m^2 around noon.
            ang = np.pi * (h - 6) / 12.0
            clear = 1000.0 * max(0.0, np.sin(ang)) if 6 <= h <= 18 else 0.0
            ghi = clear * cloud * rng.uniform(0.85, 1.05)
            rows.append((2023, 1 + d // 28, 1 + d % 28, h, 0, round(max(0.0, ghi), 1)))

    data = pd.DataFrame(rows, columns=["Year", "Month", "Day", "Hour", "Minute", "GHI"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        # Two PSM3-style metadata rows so the loader's skiprows=2 works.
        f.write("Source,Location ID,City,State,Country,Latitude,Longitude,Time Zone\n")
        f.write("DEMO,000000,Demo,NA,NA,37.0,-122.0,-8\n")
        data.to_csv(f, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NSRDB trace loader utilities")
    ap.add_argument("--make-demo-csv", type=str, default=None,
                    help="Write a fake-but-realistic PSM3 CSV for smoke testing.")
    ap.add_argument("--inspect", type=str, default=None,
                    help="Load a real CSV and print summary stats.")
    args = ap.parse_args()

    if args.make_demo_csv:
        p = synthesize_demo_csv(args.make_demo_csv)
        print(f"Wrote demo CSV: {p}")
    if args.inspect:
        s = load_ghi_series(args.inspect)
        print(f"Hours: {s.size} | mean GHI {s.mean():.1f} W/m^2 | "
              f"max {s.max():.1f} | nonzero frac {np.mean(s > 0):.2f}")
