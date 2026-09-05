import pandas as pd
import matplotlib.pyplot as plt

target = "AT2026zcl"#zcl"

df = pd.read_csv("../../../DATA/LCO-observations/"+target+"_REDUCED/lightcurve_output.csv")

filters = ["B", "V", "R", "I"]

plt.figure(figsize=(8, 6))

for f in filters:
    d = df[df["filter"] == f].copy()

    # Use PSF photometry, falling back to aperture photometry
    mag = d["mag_psf"].fillna(d["mag_ap"])
    err = d["mag_psf_err"].fillna(d["mag_ap_err"])

    good = mag.notna() & err.notna()

    plt.errorbar(
        d.loc[good, "mjd"],
        mag[good],
        yerr=err[good],
        fmt="o",
        label=f,
        capsize=3,
    )

plt.gca().invert_yaxis()
plt.xlabel("MJD")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(target+"lightcurve.png", dpi=300, bbox_inches="tight")