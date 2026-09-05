#### This file is a Milky-way.py wrapper

# it runs that file to find an EW val of some sky position and then plots that against the dust map value


from __future__ import print_function
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery


ra, dec = 156.370792, -39.830889

coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
sfd = SFDQuery()
ebv = sfd(coords)

print('E(B-V) = {:.3f} mag'.format(ebv))


# building the scatter relation between EW and Av

SNe_list = ["SN2010ev", "SN2007cq", "SN2011jm"]
AVs = []
EWs = []

#for SN_name in SNe_list:
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

"""ra, dec = 156.370792, -39.830889

    coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coords)

    print('E(B-V) = {:.3f} mag'.format(ebv))
"""


import subprocess

# Your calculation
EW = 2.4832

# Get the current Git commit
git_version = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    text=True
).strip()

print(git_version)
"""# Append result
with open("EW_results.txt", "a") as f:
    f.write(f"{EW}\t{git_version}\n")"""