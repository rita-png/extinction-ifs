#### This file is a Milky-way.py wrapper

# it runs that file to find an EW val of some sky position and then plots that against the dust map value
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

import numpy as np


ra, dec = 156.370792, -39.830889

coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
sfd = SFDQuery()
ebv = sfd(coords)

print('E(B-V) = {:.3f} mag'.format(ebv))

# building the scatter relation between EW and Av

SNe_list = ["SN2010ev", "SN2007cq", "SN2011jm", "SN2001el"]#, "ASASSN-14ad", "CSP13aao", "CSP13abl", "CSP14aaq", "LSQ12ca", "SN2019ehk"]
EBVs = []
EWs = []

for SN_name in SNe_list:
    if SN_name=="SN2010ev":
        z=0.00921
        ra, dec = 156.370792, -39.830889
    elif SN_name=="SN2007cq":
        z=0.026018
        ra, dec = 333.66965, 5.078526
    elif SN_name=="SN2019ehk":
        z=0.0043
        ra,dec=185.733958, 15.826119
    elif SN_name=="SN2011jm":
        z=0.00326
        ra,dec=193.7129, 0.6541
    elif SN_name=="SN2001el":
        z=0.003896
        ra,dec=56.12738,-44.63992

    coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coords)
    print('E(B-V) = {:.3f} mag'.format(ebv))

    EBVs.append(ebv)

EWs=[0.56, 0.40,1.55, 0.21]
EWs_errs=[0.18, 0.12, 1.52, 0.25]

# If needed, convert the corresponding E(B-V) uncertainty to the same scale as the paper relation.
# The Poznanski relation is expressed as log10(E(B-V)) = 1.17 * EW - 1.85 +/- 0.08.

plt.figure(figsize=(7, 5))
plt.xlabel("EW (Angstroms)")
plt.ylabel("E(B-V) (mag)")
plt.yscale('log')

plt.errorbar(EWs, EBVs, xerr=EWs_errs, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
plt.title("Milky-Way: EW vs E(B-V)")

# plot predicted relation from Poznanski et al. 2012
# log10(E(B-V)) = 1.17 * EW - 1.85 +/- 0.08
x = np.linspace(0, 2, 100)
log10_ebv = 1.17 * x - 1.85
y = 10 ** log10_ebv
y_lower = 10 ** (log10_ebv - 0.08)
y_upper = 10 ** (log10_ebv + 0.08)

plt.fill_between(x, y_lower, y_upper, color='red', alpha=0.15)
plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='Poznanski et al. 2012')
plt.legend()
plt.savefig("Milky-Way_EW_vs_Av.png", bbox_inches='tight')


"""# run the Milky-Way.py script for each SN
    import subprocess
    subprocess.run(["python", "Milky-Way.py", "--SN_name", SN_name])

    # read the output files to get the Av and EW values
    with open(f"{SN_name}_Av.txt", "r") as f:
        AV = float(f.read().strip())
        AVs.append(AV)

    
    with open(f"{SN_name}_EW.txt", "r") as f:
        EW = float(f.read().strip())
        EWs.append(EW)"""


    

