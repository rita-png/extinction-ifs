import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sparcl.client import SparclClient
from astropy.coordinates import SkyCoord
import astropy.units as u

import pickle
import os
from pathlib import Path


#does na id EW and DIB correlate?
#look at individual plots of EW measurements
#try also bin by sky position and not e(B-V)
#excluir bins no plot final Av vs EW com outro criterio, see quality_ok. this should be more of a percentage or relative error
#more elements
#can i get more spectra? galaxies, ELG, etc

import sys, os, importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import functions
importlib.reload(functions)
from functions import *

# ============================================================
# 1. SETTINGS
# ============================================================


# Whether to download additional spectra from SPARCL
DOWNLOAD_NEW_DATA = False  # True = download more; False = use existing cache

DATA_RELEASE = "DESI-DR1"

# Select background galaxies.
Z_MIN = 0.8
Z_MAX = 1.6

# Search the DESI footprint in 12 RA intervals.
RA_BIN_WIDTH = 30.0
DEC_RANGE = [-30.0, 85.0]

# Number of spectra to request per RA interval.
N_PER_RA_BIN = 200

# Minimum approximate continuum S/N per wavelength pixel.
MIN_SNR = 1.0

# DESI wavelength system is vacuum-barycentric.
# Measure the combined Milky Way Na I D doublet.
LINE_MIN = 5882.5
LINE_MAX = 5903.5

# Continuum sidebands.0.5
BLUE_CONT = (5868.0, 5878.0)
RED_CONT = (5908.0, 5918.0)

# Output files.
CSV_FILE = "desi_mw_nai_d_ew.csv"
PLOT_FILE = "desi_mw_nai_d_ew.png"

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# Bin width in log10(E(B-V)), measured in dex.
EBV_BIN_WIDTH = 0.15

# Minimum number of quality-approved galaxies per reddening bin.
MIN_GALAXIES_PER_EBV_BIN = 4

# Quality criteria for individual EW measurements.
MIN_SNR_FOR_BINNING = 0.0
MAX_EW_ERR_FOR_BINNING = 6.0  # Angstrom

# Diagnostic plots.
DIAGNOSTIC_DIR = Path("nai_d_diagnostics")
DIB_DIAGNOSTIC_DIR = Path("dib5780_diagnostics")
DIAGNOSTIC_DIR.mkdir(exist_ok=True)
DIB_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)


# Save spectra for the most extreme measurements in each bin.
N_SPECTRA_TO_PLOT_PER_BIN = 6

# Output files.
CSV_FILE = "desi_mw_nai_d_ew.csv"
EBV_BINS_CSV = "desi_nai_d_ebv_bins.csv"
NA_CSV = "desi_nai_d_ebv_bins.csv"
DIB_CSV = "desi_dib5780_ebv_bins.csv"
OUT_PNG = "desi_dib_vs_nai_d.png"


# Cleaning old plots
def clear_pngs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("*.png"):
        f.unlink()

clear_pngs(DIAGNOSTIC_DIR)
clear_pngs(DIB_DIAGNOSTIC_DIR)

# ============================================================
# 2. CONNECT TO PUBLIC DESI SPECTRA THROUGH SPARCL
# ============================================================


if DOWNLOAD_NEW_DATA:

    client = SparclClient(announcement=False)

    search_fields = [
        "sparcl_id",
        "ra",
        "dec",
        "redshift",
        "spectype",
        "data_release",
    ]

    base_constraints = {
        "spectype": ["GALAXY"],
        "redshift": [Z_MIN, Z_MAX],
        "data_release": [DATA_RELEASE],
    }


    spectrum_ids = []
    print("Searching DESI DR1 for background galaxies...")

    for ra_min in np.arange(0.0, 360.0, RA_BIN_WIDTH):

        ra_max = ra_min + RA_BIN_WIDTH

        constraints = dict(base_constraints)
        constraints["ra"] = [float(ra_min), float(ra_max)]
        constraints["dec"] = DEC_RANGE

        found = client.find(
            outfields=search_fields,
            constraints=constraints,
            limit=N_PER_RA_BIN,
        )

        new_ids = list(found.ids)
        spectrum_ids.extend(new_ids)

        print(
            f"RA = {ra_min:5.1f}-{ra_max:5.1f} deg: "
            f"{len(new_ids)} spectra"
        )


    # Remove any duplicate spectrum identifiers.
    spectrum_ids = list(dict.fromkeys(spectrum_ids))

    if len(spectrum_ids) == 0:
        raise RuntimeError(
            "No spectra found. Check the SPARCL service, data-release "
            "name, and search constraints."
        )

    print(f"\nTotal unique spectra found: {len(spectrum_ids)}")


# ============================================================
# SECTION 3: SPECTRA CACHE AND CONDITIONAL DOWNLOADING
# ============================================================

import os
import pickle
from pathlib import Path


# ------------------------------------------------------------
# 3.1 Cache configuration
# ------------------------------------------------------------

# Save the cache in the same folder as this Python script.
CACHE_FILE = Path(__file__).resolve().with_name(
    "desi_spectra_cache.pkl"
)

# Fields to retain from each spectrum.
CACHE_FIELDS = [
    "sparcl_id",
    "ra",
    "dec",
    "redshift",
    "spectype",
    "data_release",
    "wavelength",
    "flux",
    "ivar",
    "model",
]


# ------------------------------------------------------------
# 3.2 Helper functions
# ------------------------------------------------------------

def get_field(record, name):
    """Read a field from a SPARCL record or dictionary."""
    try:
        return getattr(record, name)
    except AttributeError:
        return record[name]


def make_plain_record(record):
    """Convert a SPARCL record into a cache-friendly dictionary."""
    return {
        field: get_field(record, field)
        for field in CACHE_FIELDS
    }


def save_cache(cache):
    """Write the cache atomically to avoid incomplete cache files."""
    temp_file = CACHE_FILE.with_name(CACHE_FILE.name + ".tmp")

    with temp_file.open("wb") as f:
        pickle.dump(
            cache,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    os.replace(temp_file, CACHE_FILE)


# ------------------------------------------------------------
# 3.3 Load the existing cache
# ------------------------------------------------------------

if CACHE_FILE.exists():

    print(f"Loading existing cache: {CACHE_FILE}")

    with CACHE_FILE.open("rb") as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict):
        raise ValueError(
            "The cache file does not contain the expected dictionary."
        )

else:

    print("No previous spectra cache found.")

    cache = {
        "spectrum_ids": [],
        "completed_ids": [],
        "records": [],
    }


# Normalize older cache structures.
cache.setdefault("spectrum_ids", [])
cache.setdefault("completed_ids", [])
cache.setdefault("records", [])


# Build a lookup of records already stored.
records_by_id = {
    str(get_field(record, "sparcl_id")): record
    for record in cache["records"]
}

# A stored record counts as completed.
completed = set(map(str, cache["completed_ids"]))
completed.update(records_by_id.keys())


# ------------------------------------------------------------
# 3.4 Decide whether to download more spectra
# ------------------------------------------------------------

if DOWNLOAD_NEW_DATA:

    print("\nDOWNLOAD_NEW_DATA = True")
    print("Checking which requested spectra still need downloading.")

    # These IDs must be produced by the SPARCL search in Section 2.
    requested_ids = list(dict.fromkeys(
        str(sid) for sid in spectrum_ids
    ))

    # Retain old IDs and add newly discovered IDs.
    all_known_ids = list(dict.fromkeys(
        list(map(str, cache["spectrum_ids"]))
        + requested_ids
    ))

    cache["spectrum_ids"] = all_known_ids

    # Only retrieve spectra that are not already cached.
    remaining_ids = [
        sid for sid in requested_ids
        if sid not in completed
    ]

    print(f"Previously cached spectra: {len(records_by_id)}")
    print(f"New spectra to retrieve: {len(remaining_ids)}")

    batch_size = 10

    # Download the missing spectra in small batches.
    for start in range(0, len(remaining_ids), batch_size):

        batch_ids = remaining_ids[start:start + batch_size]

        print(
            f"\nRetrieving batch {start // batch_size + 1}: "
            f"{len(batch_ids)} spectra..."
        )

        try:

            batch = client.retrieve(
                uuid_list=batch_ids,
                include=CACHE_FIELDS,
                limit=len(batch_ids),
                verbose=True,
            )

        except Exception as exc:

            # Save all successful work before reporting the error.
            cache["records"] = list(records_by_id.values())
            cache["completed_ids"] = list(completed)
            save_cache(cache)

            print(
                f"\nRetrieval failed: {exc}\n"
                "Previously retrieved spectra have been saved.\n"
                "You can rerun with DOWNLOAD_NEW_DATA = True "
                "to attempt the remaining downloads."
            )

            raise

        returned_ids = set()

        # Store the records SPARCL actually returned.
        for record in batch.records:

            plain = make_plain_record(record)
            sid = str(plain["sparcl_id"])

            records_by_id[sid] = plain
            returned_ids.add(sid)

        # Mark only returned records as completed.
        completed.update(returned_ids)

        # Save the updated cache after every successful batch.
        cache["records"] = list(records_by_id.values())
        cache["completed_ids"] = list(completed)

        save_cache(cache)

        print(f"Cached spectra: {len(cache['records'])}")

    print("\nDownload step finished.")

else:

    print("\nDOWNLOAD_NEW_DATA = False")
    print("Skipping additional spectrum downloads.")

    if not CACHE_FILE.exists():

        raise FileNotFoundError(
            f"No cached spectra found at:\n{CACHE_FILE.resolve()}\n"
            "Set DOWNLOAD_NEW_DATA = True to download spectra first."
        )


# ------------------------------------------------------------
# 3.5 Prepare the spectra for the rest of the analysis
# ------------------------------------------------------------

# Always use the accumulated records from the cache.
all_records = cache["records"]

if not all_records:

    raise ValueError(
        "The cache contains no spectra to analyze. "
        "Check your cache file or set DOWNLOAD_NEW_DATA = True."
    )

print(
    f"\nTotal spectra available for analysis: {len(all_records)}"
)

# ============================================================
# 4. PREPARE SPECTRA AND QUERY DUST REDDENING
# ============================================================

from dustmaps.sfd import SFDQuery
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

print("\nPreparing cached spectra for reddening-based stacking...")

# ------------------------------------------------------------
# Na I D configuration
# ------------------------------------------------------------

NAI_D2 = 5889.95
NAI_D1 = 5895.92

# Centre of the combined doublet.
NAI_D_CENTER = (NAI_D2 + NAI_D1) / 2.0

# Half-width of the EW integration window.
# This approximately reproduces the original 5882.5–5903.5 Å window.
NAI_V_WINDOW = 530.0

NAI_HALF_WIDTH = NAI_D_CENTER * NAI_V_WINDOW / 300000.0

# The EW function also needs wavelengths around the line to fit
# the continuum. Keep a 200 Å-wide region centred on Na I D.
STACK_WAVE_HALF_WIDTH = 100.0



# ------------------------------------------------------------
# Extract the spectral arrays and sky coordinates.
# Do not calculate individual EWs here.
# ------------------------------------------------------------

spectral_data = []

for i, spec in enumerate(all_records):

    try:
        wave = np.asarray(
            get_field(spec, "wavelength"),
            dtype=float
        )

        flux = np.asarray(
            get_field(spec, "flux"),
            dtype=float
        )

        ivar = np.asarray(
            get_field(spec, "ivar"),
            dtype=float
        )

        model = np.asarray(
            get_field(spec, "model"),
            dtype=float
        )

        ra = float(get_field(spec, "ra"))
        dec = float(get_field(spec, "dec"))
        redshift = float(get_field(spec, "redshift"))

    except (AttributeError, KeyError, TypeError, ValueError) as exc:

        warnings.warn(
            f"Skipping cached spectrum {i}: {exc}"
        )
        continue

    # Require matching one-dimensional spectral arrays.
    if not (
        wave.ndim == 1
        and flux.ndim == 1
        and ivar.ndim == 1
        and model.ndim == 1
        and len(wave) == len(flux) == len(ivar) == len(model)
    ):
        warnings.warn(
            f"Skipping spectrum {i}: incompatible array shapes."
        )
        continue

    if not (
        np.isfinite(ra)
        and np.isfinite(dec)
        and -90.0 <= dec <= 90.0
    ):
        continue

    # Retain the spectrum even if some wavelength pixels are invalid.
    spectral_data.append({
        "ra_deg": ra,
        "dec_deg": dec,
        "redshift": redshift,
        "wave": wave,
        "flux": flux,
        "ivar": ivar,
        "model": model,
    })

if len(spectral_data) == 0:
    raise RuntimeError(
        "No usable spectra were found in the cache."
    )

print(f"Cached spectra available for stacking: {len(spectral_data)}")


# ------------------------------------------------------------
# Query SFD E(B-V) for every retained spectrum.
# ------------------------------------------------------------

coords = SkyCoord(
    ra=np.array(
        [item["ra_deg"] for item in spectral_data]
    ) * u.deg,

    dec=np.array(
        [item["dec_deg"] for item in spectral_data]
    ) * u.deg,

    frame="icrs",
)

print("\nQuerying the SFD dust map for E(B-V)...")

sfd = SFDQuery()

ebv_values = np.asarray(
    sfd(coords),
    dtype=float
)

metadata = pd.DataFrame({
    "data_idx": np.arange(len(spectral_data)),
    "ra_deg": [item["ra_deg"] for item in spectral_data],
    "dec_deg": [item["dec_deg"] for item in spectral_data],
    "redshift": [item["redshift"] for item in spectral_data],
    "ebv": ebv_values,
    "l_deg": coords.galactic.l.deg,
    "b_deg": coords.galactic.b.deg,
})

metadata["log_ebv"] = np.nan

positive_ebv = (
    np.isfinite(metadata["ebv"])
    & (metadata["ebv"] > 0)
)

metadata.loc[positive_ebv, "log_ebv"] = np.log10(
    metadata.loc[positive_ebv, "ebv"]
)

# Non-positive or invalid E(B-V) values cannot enter logarithmic bins.
metadata = metadata[
    np.isfinite(metadata["log_ebv"])
].copy()

metadata["data_idx"] = metadata["data_idx"].astype(int)

if metadata.empty:
    raise RuntimeError(
        "No spectra have valid positive E(B-V) values."
    )

print(
    f"Spectra with valid E(B-V): {len(metadata)}"
)

print(
    f"Median E(B-V): {metadata['ebv'].median():.4f} mag"
)


# ============================================================
# 5. BUILD A COMMON WAVELENGTH GRID
# ============================================================

# Select a wavelength grid around Na I D from an available spectrum.
# Individual spectra can be interpolated onto this reference grid.

wave_ref = None

for item in spectral_data:

    wave = item["wave"]

    valid_wave = (
        np.isfinite(wave)
        & (
            np.abs(wave - NAI_D_CENTER)
            <= STACK_WAVE_HALF_WIDTH
        )
    )

    candidate = np.unique(wave[valid_wave])

    if len(candidate) >= 30:
        wave_ref = candidate
        break

if wave_ref is None:
    raise RuntimeError(
        "Could not find a suitable wavelength grid around Na I D."
    )

print(
    f"Common wavelength grid contains {len(wave_ref)} pixels."
)


# ============================================================
# 6. NORMALIZE AND INTERPOLATE EACH SPECTRUM
# ============================================================

def normalize_and_interpolate(item, target_wave):
    """
    Divide flux by the DESI model and interpolate onto target_wave.

    Propagate the supplied inverse variance through the normalization
    and interpolate the normalized-spectrum variance.

    Returns
    -------
    normalized : ndarray
        Normalized flux on the common wavelength grid.
    sigma : ndarray
        Uncertainty of normalized flux on that grid.
    """

    wave = item["wave"]
    flux = item["flux"]
    ivar = item["ivar"]
    model = item["model"]

    valid = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & np.isfinite(model)
        & (ivar > 0)
        & (model > 0)
    )

    if valid.sum() < 30:
        return None, None

    w = wave[valid]
    f = flux[valid]
    iv = ivar[valid]
    m = model[valid]

    # Normalize by the DESI best-fit model.
    normalized = f / m

    # sigma(flux/model) = 1 / (sqrt(ivar) * abs(model))
    sigma = 1.0 / (
        np.sqrt(iv) * np.abs(m)
    )

    valid_norm = (
        np.isfinite(w)
        & np.isfinite(normalized)
        & np.isfinite(sigma)
        & (sigma > 0)
    )

    w = w[valid_norm]
    normalized = normalized[valid_norm]
    sigma = sigma[valid_norm]

    if len(w) < 30:
        return None, None

    # Sort wavelength samples.
    order = np.argsort(w)

    w = w[order]
    normalized = normalized[order]
    sigma = sigma[order]

    # Ensure that wavelengths are unique.
    w, unique_indices = np.unique(
        w,
        return_index=True
    )

    normalized = normalized[unique_indices]
    sigma = sigma[unique_indices]

    if len(w) < 30:
        return None, None

    # Do not extrapolate beyond the actual wavelength coverage.
    inside = (
        (target_wave >= w[0])
        & (target_wave <= w[-1])
    )

    normalized_out = np.full(
        target_wave.shape,
        np.nan,
        dtype=float
    )

    sigma_out = np.full(
        target_wave.shape,
        np.nan,
        dtype=float
    )

    normalized_out[inside] = np.interp(
        target_wave[inside],
        w,
        normalized
    )

    # Interpolate variance rather than standard deviation.
    variance = sigma**2

    sigma_out[inside] = np.sqrt(
        np.interp(
            target_wave[inside],
            w,
            variance
        )
    )

    # Require reasonable wavelength coverage across the doublet.
    line_window = (
        np.abs(target_wave - NAI_D_CENTER)
        <= NAI_HALF_WIDTH
    )

    n_line_pixels = int(line_window.sum())

    if n_line_pixels == 0:
        return None, None

    n_valid_line = np.sum(
        np.isfinite(normalized_out[line_window])
        & np.isfinite(sigma_out[line_window])
        & (sigma_out[line_window] > 0)
    )

    if n_valid_line < 0.90 * n_line_pixels:
        return None, None

    return normalized_out, sigma_out


# Project all usable individual spectra onto the common grid.
# No individual EW or individual S/N threshold is applied here.

projected = {}

print("\nNormalizing and interpolating spectra...")

for j, item in enumerate(spectral_data):

    normalized, sigma = normalize_and_interpolate(
        item,
        wave_ref
    )

    if normalized is not None:
        projected[j] = (normalized, sigma)

    if (j + 1) % 200 == 0 or (j + 1) == len(spectral_data):
        print(
            f"Prepared {j + 1}/{len(spectral_data)} spectra; "
            f"{len(projected)} currently usable."
        )

if len(projected) == 0:
    raise RuntimeError(
        "No spectra have sufficient wavelength coverage around Na I D."
    )

# Restrict the metadata to spectra with usable projected data.
metadata = metadata[
    metadata["data_idx"].isin(projected.keys())
].copy()

print(
    f"\nSpectra available for stacking: {len(metadata)}"
)


# ============================================================
# 7. ASSIGN SPECTRA TO LOG10(E(B-V)) BINS
# ============================================================

# Use the bin width already defined in Section 1.
# EBV_BIN_WIDTH is in dex.

minimum_log_ebv = metadata["log_ebv"].min()
maximum_log_ebv = metadata["log_ebv"].max()

bin_lower = (
    np.floor(minimum_log_ebv / EBV_BIN_WIDTH)
    * EBV_BIN_WIDTH
)

bin_upper = (
    np.floor(maximum_log_ebv / EBV_BIN_WIDTH) + 1
) * EBV_BIN_WIDTH

# Ensure the upper edge is strictly above the largest value.
if bin_upper <= maximum_log_ebv:
    bin_upper += EBV_BIN_WIDTH

edges = np.arange(
    bin_lower,
    bin_upper + EBV_BIN_WIDTH * 0.5,
    EBV_BIN_WIDTH
)

metadata["ebv_bin"] = pd.cut(
    metadata["log_ebv"],
    bins=edges,
    right=False,
    include_lowest=True,
)

print(f"\nRequested log10(E(B-V)) bin width: {EBV_BIN_WIDTH} dex")


# ============================================================
# 8. STACK SPECTRA WITHIN EACH REDDENING BIN
# ============================================================

stacked_spectra = []
stacked_errors = []
stack_descriptions = []

for interval, group in metadata.groupby(
    "ebv_bin",
    observed=True,
    sort=True,
):

    if pd.isna(interval):
        continue

    # Keep the spectra that have usable wavelength coverage.
    indices = [
        int(j)
        for j in group["data_idx"]
        if int(j) in projected
    ]

    n_spectra = len(indices)

    # Do not create a stack from too few spectra.
    if n_spectra < MIN_GALAXIES_PER_EBV_BIN:

        print(
            f"Skipping bin {interval}: "
            f"only {n_spectra} spectra "
            f"(minimum {MIN_GALAXIES_PER_EBV_BIN})."
        )

        continue

    # Inverse-variance-weighted coaddition.
    #
    # stack(lambda) = sum[flux_norm / sigma_norm^2]
    #                 / sum[1 / sigma_norm^2]
    #
    # sigma_stack(lambda) = sqrt(1 / sum[1 / sigma_norm^2])

    numerator = np.zeros(
        len(wave_ref),
        dtype=float
    )

    denominator = np.zeros(
        len(wave_ref),
        dtype=float
    )

    for j in indices:

        normalized, sigma = projected[j]

        valid = (
            np.isfinite(normalized)
            & np.isfinite(sigma)
            & (sigma > 0)
        )

        weights = np.zeros(
            len(wave_ref),
            dtype=float
        )

        weights[valid] = 1.0 / sigma[valid]**2

        numerator[valid] += (
            normalized[valid] * weights[valid]
        )

        denominator[valid] += weights[valid]

    stack = np.full(
        len(wave_ref),
        np.nan,
        dtype=float
    )

    stack_error = np.full(
        len(wave_ref),
        np.nan,
        dtype=float
    )

    valid_stack = denominator > 0

    stack[valid_stack] = (
        numerator[valid_stack]
        / denominator[valid_stack]
    )

    stack_error[valid_stack] = (
        1.0 / np.sqrt(denominator[valid_stack])
    )

    # Require sufficient valid pixels around the doublet.
    line_window = (
        np.abs(wave_ref - NAI_D_CENTER)
        <= NAI_HALF_WIDTH
    )

    valid_line_stack = (
        np.isfinite(stack[line_window])
        & np.isfinite(stack_error[line_window])
        & (stack_error[line_window] > 0)
    )

    if valid_line_stack.sum() < 0.90 * line_window.sum():

        print(
            f"Skipping bin {interval}: "
            "insufficient valid pixels in stacked spectrum."
        )

        continue

    members = group[
        group["data_idx"].isin(indices)
    ].copy()

    title = (
        f"log10(E(B-V)) = "
        f"[{interval.left:.2f}, {interval.right:.2f}); "
        f"N={n_spectra}"
    )

    stacked_spectra.append(stack)
    stacked_errors.append(stack_error)

    stack_descriptions.append({
        "interval": interval,
        "members": members,
        "n_spectra": n_spectra,
        "title": title,
    })

    print(
        f"Stacked bin {interval}: "
        f"{n_spectra} spectra."
    )

if len(stacked_spectra) == 0:
    raise RuntimeError(
        "No reddening bins contain enough usable spectra. "
        "Consider increasing EBV_BIN_WIDTH or reducing "
        "MIN_GALAXIES_PER_EBV_BIN."
    )

stacked_spectra = np.asarray(stacked_spectra)
stacked_errors = np.asarray(stacked_errors)

print(
    f"\nTotal stacked spectra: {len(stacked_spectra)}"
)


# ============================================================
# 9. MEASURE EW USING YOUR EW_VORONOI_BINS FUNCTION
# ============================================================

print("\nMeasuring Na I D EW from the stacked spectra...")

"""Simple equivalent-width measurement for stacked spectra.

Continuum: weighted polynomial fit (default straight line) to two
sidebands. EW: sum of (1 - flux/continuum) * dlambda over the line window.
Error: propagated pixel errors + continuum-fit covariance, inflated by
sqrt(reduced chi2) of the sideband fit when that exceeds 1.
Positive EW = absorption.
"""


def ew_fixed_window(wave, spectra, errors, center, half_width, min_cover=0.9):
    """EW = sum((1 - flux) * dlambda) over |wave - center| <= half_width.
    Assumes the continuum is exactly 1. The error contains pixel noise
    only (no continuum uncertainty). Returns (ew, ew_err)."""
    wave = np.asarray(wave, float)
    spectra = np.atleast_2d(np.asarray(spectra, float))
    errors = np.atleast_2d(np.asarray(errors, float))
    win = np.abs(wave - center) <= half_width
    dlam = np.gradient(wave)[win]
    f, s = spectra[:, win], errors[:, win]
    ok = np.isfinite(f) & np.isfinite(s) & (s > 0)
    enough = ok.mean(axis=1) >= min_cover
    ew = np.nansum(np.where(ok, (1.0 - f) * dlam, 0.0), axis=1)
    err = np.sqrt(np.nansum(np.where(ok, (dlam * s) ** 2, 0.0), axis=1))
    return np.where(enough, ew, np.nan), np.where(enough, err, np.nan)

    

EW_stacked, EW_stacked_err = ew_fixed_window(
    wave_ref, stacked_spectra, stacked_errors,
    center=NAI_D_CENTER, half_width=NAI_HALF_WIDTH
)

# ============================================================
# 10. CREATE THE BIN-LEVEL RESULTS TABLE
# ============================================================

bin_results = []

for i, description in enumerate(stack_descriptions):

    interval = description["interval"]
    members = description["members"]

    log_ebv_values = members["log_ebv"].to_numpy()

    mean_log_ebv = np.mean(log_ebv_values)

    if len(log_ebv_values) > 1:
        sem_log_ebv = (
            np.std(log_ebv_values, ddof=1)
            / np.sqrt(len(log_ebv_values))
        )
    else:
        sem_log_ebv = np.nan

    ew = EW_stacked[i]
    ew_err = EW_stacked_err[i]

    # Flag uncertain stacks rather than discarding their results.
    # Low-quality bins will appear gray in the final plot.
    quality_ok = (
        np.isfinite(ew)
        and np.isfinite(ew_err)
        and ew_err > 0
        and ew_err <= MAX_EW_ERR_FOR_BINNING
    )

    bin_results.append({
        "ebv_bin": str(interval),
        "log_ebv_low": float(interval.left),
        "log_ebv_high": float(interval.right),
        "n_spectra": description["n_spectra"],
        "mean_ebv": members["ebv"].mean(),
        "median_ebv": members["ebv"].median(),
        "mean_log_ebv": mean_log_ebv,
        "sem_log_ebv": sem_log_ebv,
        "ew_A": ew,
        "ew_err_A": ew_err,
        "quality_ok": quality_ok,
    })

bin_df = pd.DataFrame(bin_results)

# Save the stacked-bin measurements.
bin_df.to_csv(
    EBV_BINS_CSV,
    index=False
)

print("\n" + "=" * 55)
print("STACKED Na I D MEASUREMENTS")
print("=" * 55)

print(f"Spectra available for stacking: {len(metadata)}")
print(f"Reddening bins with stacks: {len(bin_df)}")

print(
    f"Stacks passing the EW uncertainty cut: "
    f"{bin_df['quality_ok'].sum()}"
)

print(f"Saved results to: {EBV_BINS_CSV}")

print(
    bin_df[
        [
            "ebv_bin",
            "n_spectra",
            "median_ebv",
            "ew_A",
            "ew_err_A",
            "quality_ok",
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. PLOT EACH STACKED SPECTRUM AROUND Na I D
# ============================================================

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plot_window = (
    (wave_ref >= 5865.0)
    & (wave_ref <= 5920.0)
)

for i, description in enumerate(stack_descriptions):

    interval = description["interval"]

    stack = stacked_spectra[i]
    stack_error = stacked_errors[i]

    w = wave_ref[plot_window]
    f = stack[plot_window]
    err = stack_error[plot_window]

    valid = (
        np.isfinite(w)
        & np.isfinite(f)
        & np.isfinite(err)
    )

    if valid.sum() < 10:
        continue

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(
        w[valid],
        f[valid] - err[valid],
        f[valid] + err[valid],
        alpha=0.2,
        label="Stacked-spectrum uncertainty",
    )

    ax.plot(
        w[valid],
        f[valid],
        linewidth=1,
        label="Inverse-variance-weighted stack",
    )

    ax.axvline(
        NAI_D2,
        linestyle="--",
        linewidth=1,
        label="Na I D2",
    )

    ax.axvline(
        NAI_D1,
        linestyle=":",
        linewidth=1,
        label="Na I D1",
    )

    ax.axvspan(
        NAI_D_CENTER - NAI_HALF_WIDTH,
        NAI_D_CENTER + NAI_HALF_WIDTH,
        alpha=0.10,
        label="EW integration window",
    )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel("Observed wavelength [Å]")
    ax.set_ylabel("Flux / DESI model")

    ax.set_title(
        f"{description['title']}\n"
        f"EW = {EW_stacked[i]:.3f} ± "
        f"{EW_stacked_err[i]:.3f} Å"
    )

    ax.legend(fontsize=8)
    plt.tight_layout()

    safe_name = (
        f"stack_ebv_{interval.left:.2f}_"
        f"to_{interval.right:.2f}.png"
    ).replace("-", "m").replace(".", "p")

    fig.savefig(
        DIAGNOSTIC_DIR / safe_name,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

print(
    f"\nStacked-spectrum plots saved in: "
    f"{DIAGNOSTIC_DIR.resolve()}"
)


# ============================================================
# 12. PLOT STACKED Na I D EW AGAINST SFD E(B-V)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Separate higher-quality and uncertain measurements.
good = bin_df[bin_df["quality_ok"]].copy()
uncertain = bin_df[~bin_df["quality_ok"]].copy()

# Higher-quality measurements with error bars.
ax.errorbar(
    good["ew_A"],
    good["mean_log_ebv"],
    xerr=good["ew_err_A"],
    yerr=good["sem_log_ebv"],
    fmt="o",
    color="tab:blue",
    ecolor="tab:blue",
    capsize=3,
    markersize=6,
    alpha=0.85,
    label="Higher-quality stacks",
)

# Uncertain measurements without error bars.
if not uncertain.empty:
    ax.scatter(
        uncertain["ew_A"],
        uncertain["mean_log_ebv"],
        color="gray",
        s=45,
        alpha=0.8,
        label="Uncertain stacks (error bars omitted)",
    )

# Poznanski et al. (2012):
# log10(E(B-V)) = 1.17 * EW(Na I D) - 1.85

finite = bin_df[
    np.isfinite(bin_df["ew_A"])
    & np.isfinite(bin_df["mean_log_ebv"])
]

if not finite.empty:

    x_min = min(0.0, finite["ew_A"].min() - 0.2)
    x_max = max(1.2, finite["ew_A"].max() + 0.2)

    ew_grid = np.linspace(x_min, x_max, 300)

    ax.plot(
        ew_grid,
        1.17 * ew_grid - 1.85,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Poznanski et al. (2012)",
    )

ax.set_xlabel(r"Stacked EW(Na I D) [$\mathrm{\AA}$]")
ax.set_ylabel(r"Mean $\log_{10}(E(B-V))$")

ax.set_title("Milky Way Na I D absorption versus dust reddening")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)

plt.tight_layout()

fig.savefig(
    "desi_nai_d_poznanski_stacked.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print("Na I D plot saved to desi_nai_d_poznanski_stacked.png")


# ============================================================
# 13. DIB 5780: STACK SPECTRA BY E(B-V), MEASURE EW,
#     AND COMPARE WITH LITERATURE RELATIONS
# ============================================================

DIB_CENTER = 5780.6            # Milky Way DIB wavelength [Angstrom]
DIB_GRID_HALF_WIDTH = 100.0    # Wavelength region for continuum fitting
DIB_V_WINDOW = 250.0           # EW integration half-width in velocity [km/s]

DIB_MIN_SPECTRA_PER_BIN = MIN_GALAXIES_PER_EBV_BIN
DIB_MAX_EW_ERR_A = 25        # Flag larger EW errors [Angstrom]

DIB_OUTPUT_CSV = "desi_dib5780_ebv_bins.csv"
DIB_PLOT_FILE = "desi_dib5780_reddening_relation.png"

print("\n" + "=" * 60)
print("DIB 5780 ANALYSIS")
print("=" * 60)


# ============================================================
# 13.1 BUILD A WAVELENGTH GRID AROUND DIB 5780
# ============================================================

# Reuse the reddening metadata from the existing analysis.
# data_idx refers to the corresponding entry in spectral_data.

dib_metadata = metadata.copy()

dib_wave_ref = None

for idx in dib_metadata["data_idx"].astype(int):

    wave = spectral_data[idx]["wave"]

    region = (
        np.isfinite(wave)
        & (wave >= DIB_CENTER - DIB_GRID_HALF_WIDTH)
        & (wave <= DIB_CENTER + DIB_GRID_HALF_WIDTH)
    )

    candidate = np.unique(wave[region])

    if len(candidate) >= 30:
        dib_wave_ref = candidate
        break

if dib_wave_ref is None:
    raise RuntimeError(
        "Could not construct a wavelength grid around DIB 5780. "
        "Check the wavelength coverage of the cached spectra."
    )

print(f"DIB wavelength grid: {len(dib_wave_ref)} pixels")


# Integration window used by EW_voronoi_bins.
dib_half_width = DIB_CENTER * DIB_V_WINDOW / 300000.0

dib_line_region = (
    np.abs(dib_wave_ref - DIB_CENTER) <= dib_half_width
)

if dib_line_region.sum() < 5:
    raise RuntimeError(
        "Too few wavelength pixels in the DIB integration window."
    )


# ============================================================
# 13.2 NORMALIZE AND INTERPOLATE SPECTRA
# ============================================================

def prepare_dib_spectrum(item, target_wave):
    """
    Normalize observed flux by the DESI model and interpolate
    flux and uncertainty onto the DIB wavelength grid.
    """

    wave = np.asarray(item["wave"], dtype=float)
    flux = np.asarray(item["flux"], dtype=float)
    ivar = np.asarray(item["ivar"], dtype=float)
    model = np.asarray(item["model"], dtype=float)

    valid = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & np.isfinite(model)
        & (ivar > 0)
        & (model > 0)
    )

    if valid.sum() < 30:
        return None, None

    w = wave[valid]
    normalized = flux[valid] / model[valid]

    sigma = 1.0 / (
        np.sqrt(ivar[valid]) * np.abs(model[valid])
    )

    good = (
        np.isfinite(w)
        & np.isfinite(normalized)
        & np.isfinite(sigma)
        & (sigma > 0)
    )

    w = w[good]
    normalized = normalized[good]
    sigma = sigma[good]

    if len(w) < 30:
        return None, None

    # np.interp requires ascending wavelengths.
    order = np.argsort(w)
    w = w[order]
    normalized = normalized[order]
    sigma = sigma[order]

    # Remove duplicate wavelengths.
    w, unique_indices = np.unique(w, return_index=True)
    normalized = normalized[unique_indices]
    sigma = sigma[unique_indices]

    if len(w) < 30:
        return None, None

    # Interpolate only inside the actual wavelength coverage.
    inside = (
        (target_wave >= w[0])
        & (target_wave <= w[-1])
    )

    flux_out = np.full(target_wave.shape, np.nan)
    sigma_out = np.full(target_wave.shape, np.nan)

    flux_out[inside] = np.interp(
        target_wave[inside], w, normalized
    )

    # Interpolate variance, then take its square root.
    sigma_out[inside] = np.sqrt(
        np.interp(
            target_wave[inside],
            w,
            sigma**2,
        )
    )

    # Require at least 90% coverage of the DIB integration window.
    line_region = (
        np.abs(target_wave - DIB_CENTER) <= dib_half_width
    )

    n_line = line_region.sum()

    if n_line == 0:
        return None, None

    valid_line = (
        np.isfinite(flux_out[line_region])
        & np.isfinite(sigma_out[line_region])
        & (sigma_out[line_region] > 0)
    )

    if valid_line.sum() < 0.90 * n_line:
        return None, None

    return flux_out, sigma_out


dib_projected = {}

for idx in dib_metadata["data_idx"].astype(int).unique():

    flux_norm, sigma_norm = prepare_dib_spectrum(
        spectral_data[idx],
        dib_wave_ref,
    )

    if flux_norm is not None:
        dib_projected[idx] = (flux_norm, sigma_norm)

print(
    f"Spectra usable around DIB 5780: {len(dib_projected)}"
)

dib_metadata = dib_metadata[
    dib_metadata["data_idx"].isin(dib_projected.keys())
].copy()

if dib_metadata.empty:
    raise RuntimeError(
        "No spectra have usable wavelength coverage around DIB 5780."
    )


# ============================================================
# 13.3 STACK SPECTRA IN THE EXISTING E(B-V) BINS
# ============================================================

dib_stacks = []
dib_stack_errors = []
dib_stack_info = []

for interval, group in dib_metadata.groupby(
    "ebv_bin",
    observed=True,
    sort=True,
):

    indices = group["data_idx"].astype(int).tolist()

    if len(indices) < DIB_MIN_SPECTRA_PER_BIN:
        print(
            f"Skipping DIB bin {interval}: "
            f"only {len(indices)} spectra."
        )
        continue

    numerator = np.zeros(len(dib_wave_ref))
    denominator = np.zeros(len(dib_wave_ref))

    for idx in indices:

        flux_norm, sigma_norm = dib_projected[idx]

        valid = (
            np.isfinite(flux_norm)
            & np.isfinite(sigma_norm)
            & (sigma_norm > 0)
        )

        weights = np.zeros(len(dib_wave_ref))
        weights[valid] = 1.0 / sigma_norm[valid]**2

        numerator[valid] += flux_norm[valid] * weights[valid]
        denominator[valid] += weights[valid]

    stack = np.full(len(dib_wave_ref), np.nan)
    stack_error = np.full(len(dib_wave_ref), np.nan)

    valid_stack = denominator > 0

    stack[valid_stack] = (
        numerator[valid_stack] / denominator[valid_stack]
    )

    stack_error[valid_stack] = (
        1.0 / np.sqrt(denominator[valid_stack])
    )

    valid_line = (
        np.isfinite(stack[dib_line_region])
        & np.isfinite(stack_error[dib_line_region])
        & (stack_error[dib_line_region] > 0)
    )

    if valid_line.sum() < 0.90 * dib_line_region.sum():
        print(f"Skipping bin {interval}: insufficient DIB coverage.")
        continue

    dib_stacks.append(stack)
    dib_stack_errors.append(stack_error)

    dib_stack_info.append({
        "interval": interval,
        "members": group.copy(),
        "n_spectra": len(indices),
    })

    print(f"Stacked DIB bin {interval}: {len(indices)} spectra.")

if len(dib_stacks) == 0:
    raise RuntimeError(
        "No DIB stacks were produced. Try wider E(B-V) bins "
        "or check the wavelength coverage."
    )

dib_stacks = np.asarray(dib_stacks)
dib_stack_errors = np.asarray(dib_stack_errors)

print(f"\nTotal DIB stacks: {len(dib_stacks)}")


# ============================================================
# 13.4 MEASURE DIB 5780 EWs USING YOUR FUNCTION
# ============================================================

print("\nMeasuring DIB 5780 equivalent widths...")

dib_ew, dib_ew_err = ew_fixed_window(
    dib_wave_ref, dib_stacks, dib_stack_errors,
    center=DIB_CENTER, half_width=dib_half_width
)

dib_ew = np.asarray(dib_ew, dtype=float)
dib_ew_err = np.asarray(dib_ew_err, dtype=float)

if not (
    len(dib_ew)
    == len(dib_ew_err)
    == len(dib_stack_info)
):
    raise RuntimeError(
        "EW_voronoi_bins returned an unexpected number of DIB measurements."
    )


# ============================================================
# 13.5 SAVE THE DIB BIN RESULTS
# ============================================================

dib_results = []

for i, info in enumerate(dib_stack_info):

    members = info["members"]

    mean_ebv = members["ebv"].mean()
    n = len(members)

    if n > 1:
        sem_ebv = (
            members["ebv"].std(ddof=1) / np.sqrt(n)
        )
    else:
        sem_ebv = np.nan

    ew = dib_ew[i]
    ew_err = dib_ew_err[i]

    # Keep all measurements, but flag uncertain results.
    significance = (
        ew / ew_err
        if np.isfinite(ew_err) and ew_err > 0
        else np.nan
    )

    quality_ok = (
        np.isfinite(ew)
        and np.isfinite(ew_err)
        and ew_err > 0
        and ew_err <= DIB_MAX_EW_ERR_A
        and np.isfinite(significance)
        and significance >= 2.0
    )

    dib_results.append({
        "ebv_bin": str(info["interval"]),
        "log_ebv_low": info["interval"].left,
        "log_ebv_high": info["interval"].right,
        "n_spectra": info["n_spectra"],
        "mean_ebv": mean_ebv,
        "sem_ebv": sem_ebv,
        "ew_dib5780_A": ew,
        "ew_dib5780_err_A": ew_err,
        "ew_significance": significance,
        "quality_ok": quality_ok,
    })

dib_df = pd.DataFrame(dib_results)

dib_df.to_csv(DIB_OUTPUT_CSV, index=False)

print("\nDIB 5780 measurements:")
print(
    dib_df[
        [
            "ebv_bin",
            "n_spectra",
            "mean_ebv",
            "ew_dib5780_A",
            "ew_dib5780_err_A",
            "ew_significance",
            "quality_ok",
        ]
    ].to_string(index=False)
)

print(f"\nSaved DIB table to {DIB_OUTPUT_CSV}")


# ============================================================
# 13.6 PLOT STACKED DIB SPECTRA
# ============================================================

plot_region = (
    (dib_wave_ref >= DIB_CENTER - 15)
    & (dib_wave_ref <= DIB_CENTER + 15)
)

for i, info in enumerate(dib_stack_info):

    w = dib_wave_ref[plot_region]
    f = dib_stacks[i][plot_region]
    e = dib_stack_errors[i][plot_region]

    valid = (
        np.isfinite(w)
        & np.isfinite(f)
        & np.isfinite(e)
    )

    if valid.sum() < 5:
        continue

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(
        w[valid],
        f[valid] - e[valid],
        f[valid] + e[valid],
        alpha=0.2,
        label="Stack uncertainty",
    )

    ax.plot(
        w[valid],
        f[valid],
        linewidth=1,
        label="Stacked spectrum",
    )

    ax.axvline(
        DIB_CENTER,
        linestyle="--",
        label="DIB 5780.6",
    )

    ax.axvspan(
        DIB_CENTER - dib_half_width,
        DIB_CENTER + dib_half_width,
        alpha=0.12,
        label="EW integration window",
    )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Vacuum wavelength [Å]")
    ax.set_ylabel("Observed flux / DESI model")
    ax.set_title(
        f"DIB 5780 | {info['interval']} | "
        f"N={info['n_spectra']}\n"
        f"EW = {dib_ew[i]:.4f} ± {dib_ew_err[i]:.4f} Å"
    )

    ax.legend(fontsize=8)
    plt.tight_layout()

    safe = (
        f"dib5780_{info['interval'].left:.2f}_"
        f"to_{info['interval'].right:.2f}.png"
    ).replace("-", "m").replace(".", "p")

    fig.savefig(
        DIB_DIAGNOSTIC_DIR / safe,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

print(f"DIB diagnostic spectra saved in {DIB_DIAGNOSTIC_DIR.resolve()}")


# ============================================================
# 13.7 PLOT DIB EW AGAINST E(B-V) AND LITERATURE RELATIONS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Plot higher-quality DIB measurements with error bars.
good = dib_df[dib_df["quality_ok"]].copy()

ax.errorbar(
    good["mean_ebv"],
    good["ew_dib5780_A"],
    xerr=good["sem_ebv"],
    yerr=good["ew_dib5780_err_A"],
    fmt="o",
    color="tab:blue",
    ecolor="tab:blue",
    capsize=3,
    markersize=6,
    alpha=0.85,
    label="Higher-quality stacks",
)

# Plot uncertain DIB measurements as gray points, without error bars.
uncertain = dib_df[~dib_df["quality_ok"]].copy()

if not uncertain.empty:
    ax.scatter(
        uncertain["mean_ebv"],
        uncertain["ew_dib5780_A"],
        color="gray",
        marker="o",
        s=45,
        alpha=0.8,
        zorder=3,
        label="Uncertain stacks (error bars omitted)",
    )

# Literature relations use EW in Angstrom and E(B-V) in magnitudes.
ebv_max = max(0.1, dib_df["mean_ebv"].max() * 1.10)
ebv_grid = np.linspace(0, ebv_max, 300)

# Baron et al. (2015): empirical relation from stacked SDSS spectra.
ew_baron = 0.813 * ebv_grid - 0.0032

# Friedman et al. (2011): traditional linear comparison.
ew_friedman = 0.505 * ebv_grid

ax.plot(
    ebv_grid,
    ew_baron,
    linestyle="--",
    color="black",
    linewidth=1.6,
    label="Baron et al. (2015): 0.813 E(B−V) − 0.0032",
)

ax.plot(
    ebv_grid,
    ew_friedman,
    linestyle=":",
    color="tab:red",
    linewidth=1.6,
    label="Friedman et al. (2011): 0.505 E(B−V)",
)

# Legend handles for the data-quality categories.
from matplotlib.lines import Line2D

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="tab:blue",
        linestyle="None",
        label="Higher-quality stack",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="gray",
        linestyle="None",
        label="Uncertain stack",
    ),
]

handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles + legend_handles,
    labels + ["Higher-quality stack", "Uncertain stack"],
    fontsize=8,
)

ax.set_xlabel(r"SFD $E(B-V)$ [mag]")
ax.set_ylabel(r"EW(DIB 5780.6) [$\mathrm{\AA}$]")
ax.set_title("Milky Way DIB 5780 versus dust reddening")
ax.grid(alpha=0.25)

plt.tight_layout()

fig.savefig(
    DIB_PLOT_FILE,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"\nFinal DIB 5780 comparison saved to {DIB_PLOT_FILE}")



# ============================================================
# 13.7 PLOT DIB EW AGAINST Na I D EW AND COMPUTE CORRELATIONS
# ============================================================



 
# Drop bins too noisy to carry information (set to np.inf to keep all).
MAX_NA_ERR = 3.0     # Angstrom
MAX_DIB_ERR = 0.10   # Angstrom
 
na = pd.read_csv(NA_CSV)
dib = pd.read_csv(DIB_CSV)
 
# Both tables use the same E(B-V) bins, so merge on the bin label.
df = na.merge(dib, on="ebv_bin", suffixes=("_na", "_dib"))
df = df.rename(columns={
    "ew_A": "ew_na", "ew_err_A": "ew_na_err",
    "ew_dib5780_A": "ew_dib", "ew_dib5780_err_A": "ew_dib_err",
    "mean_ebv_na": "ebv",
})
df = df[np.isfinite(df[["ew_na", "ew_na_err", "ew_dib", "ew_dib_err", "ebv"]]).all(axis=1)]
df = df[(df["ew_na_err"] <= MAX_NA_ERR) & (df["ew_dib_err"] <= MAX_DIB_ERR)].copy()
if len(df) < 3:
    raise SystemExit(f"Only {len(df)} usable bins; relax MAX_NA_ERR / MAX_DIB_ERR.")
 
# Statistics
r, p_r = stats.pearsonr(df["ew_na"], df["ew_dib"])
rho, p_rho = stats.spearmanr(df["ew_na"], df["ew_dib"])
 
# Partial correlation: remove the linear trend with E(B-V) from both EWs.
def resid(y):
    return y - np.polyval(np.polyfit(df["ebv"], y, 1), df["ebv"])
r_part, p_part = stats.pearsonr(resid(df["ew_na"]), resid(df["ew_dib"]))
 
# Weighted straight line, weights from the DIB errors (x errors ignored).
slope, icpt = np.polyfit(df["ew_na"], df["ew_dib"], 1, w=1 / df["ew_dib_err"])
 
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
ax.errorbar(df["ew_na"], df["ew_dib"], xerr=df["ew_na_err"], yerr=df["ew_dib_err"],
            fmt="none", ecolor="gray", alpha=0.6, zorder=1)
sc = ax.scatter(df["ew_na"], df["ew_dib"], c=df["ebv"], cmap="viridis", s=60, zorder=2)
xx = np.linspace(df["ew_na"].min(), df["ew_na"].max(), 100)
ax.plot(xx, slope * xx + icpt, "k--", lw=1, label=f"weighted fit, slope = {slope:.3f}")
fig.colorbar(sc, ax=ax, label="mean E(B-V) of bin [mag]")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Stacked EW(Na I D) [Å]")
ax.set_ylabel("Stacked EW(DIB 5780) [Å]")
ax.set_title("DIB 5780 vs Na I D, one point per E(B-V) bin")
ax.text(0.03, 0.97,
        f"N bins = {len(df)}\n"
        f"Pearson r = {r:.2f} (p = {p_r:.2g})\n"
        f"Spearman ρ = {rho:.2f} (p = {p_rho:.2g})\n"
        f"partial r | E(B-V) = {r_part:.2f} (p = {p_part:.2g})",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
ax.legend(loc="lower right", fontsize=8)
 
# Right panel: both EWs against E(B-V), to show whether a common trend drives it.
ax2.errorbar(df["ebv"], df["ew_na"], yerr=df["ew_na_err"], fmt="o", color="tab:blue", label="Na I D (left axis)")
ax2.set_xlabel("Mean SFD E(B-V) [mag]")
ax2.set_ylabel("EW(Na I D) [Å]", color="tab:blue")
ax3 = ax2.twinx()
ax3.errorbar(df["ebv"], df["ew_dib"], yerr=df["ew_dib_err"], fmt="s", color="tab:red", label="DIB 5780 (right axis)")
ax3.set_ylabel("EW(DIB 5780) [Å]", color="tab:red")
ax2.set_title("Both versus E(B-V)")
ax2.grid(alpha=0.25)
 
plt.tight_layout()
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.show()
print(df[["ebv_bin", "ebv", "ew_na", "ew_na_err", "ew_dib", "ew_dib_err"]].to_string(index=False))
print(f"Saved {OUT_PNG}")
