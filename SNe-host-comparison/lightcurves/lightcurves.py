from autophot import AutomatedPhotometry
from pathlib import Path

from astropy.io import fits

import os

os.chdir("/home/ritapsantos/PhD/extinction-ifs/SNe-host-comparison/lightcurves")

from pathlib import Path


# Load default configuration
config = AutomatedPhotometry.load()

# Set basic parameters
target = "AT2026wkg"
config["target_ra"] = 4.70790
config["target_dec"] = -10.36702
"""target = "AT2026zcl"
config["target_ra"] = 351.9842
config["target_dec"] = 8.7794"""

config["fits_dir"] = "../../../DATA/LCO-observations/"+target
config["target_name"] = target

# setting catalog to use Gaia for photometry
config["catalog"]["use_catalog"] = "gaia"



# changed working dir!
config["wdir"] = str(Path.cwd())

##

"""import check

h = check.  get_header("../../../DATA/LCO-observations/AT2026zcl/cpt1m010-fa14-20260826-0134-e91.fits.fz")

for k in ["TELESCOP", "INSTRUME", "FILTER", "GAIN", "RDNOISE", "SATURATE", "AIRMASS", "MJD-OBS", "DATE-OBS", "EXPTIME", "PIXSCALE"]:
    print(k, "=", h.get(k))
"""







# print basic file info
from pathlib import Path
from astropy.io import fits

fits_dir = Path("../../../DATA/LCO-observations/"+target)



########


"""import sys
sys.exit()"""


for f in fits_dir.glob("*.fits.fz"):
    with fits.open(f) as hdul:
        h = hdul[1].header

        print(
            f"{f.name:60s} "
            f"{h.get('TELESCOP'):10s} + "
            f"{h.get('INSTRUME'):6s} | "
            f"{h.get('FILTER'):4s} | "
            f"{h.get('PIXSCALE')}"
        )



# Run photometry
output_file = AutomatedPhotometry.run_photometry(default_input=config)
print(f"Results saved to: {output_file}")