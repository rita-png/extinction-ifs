
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



from importlib import resources
from powerbin import PowerBin


import sys, os, importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import functions
importlib.reload(functions)
from functions import *


SN_name="SN2010ev"#"SN2010ev"#

if SN_name=="SN2010ev":
    z=0.00921
    ra, dec = 156.370792, -39.830889
elif SN_name=="SN2007cq":
    z=0.026018
    ra, dec = 333.66965, 5.078526
elif SN_name=="SN2007bm":
    z=0.006298
    ra, dec= 171.26039,	-9.795445

file_name="DATA/"+SN_name+"/"+SN_name+".fits"


data = fits.open(file_name)
cube = data[1].data   # this is the cube, a (3681 x 341 x 604) matrix with fluxes at different 3681 wavelengths and 308 x 318 spatial pixels ("spaxels")
header = data[1].header # this has information on the data cube
ecube = data[2].data # this is the cube uncertainty (3681 x 341 x 604)




x_len=len(cube[0][0])
y_len=len(cube[0])



#  -- following is to the get the wavelength array
CRVAL = float(header["CRVAL3"])
NAXIS = int(header["NAXIS3"])
CDELT = float(header["CD3_3"])
CRPIX = float(header["CRPIX3"])
wave = np.array(CRVAL + CDELT * (np.arange(NAXIS) - CRPIX))



na_rest=(5890+5896)/2

index=findWavelengths(wave, na_rest)[1]

## Voronoi binning ##
y_center=int(y_len/2)
x_center=int(x_len/2)
region=cube[:,y_center-100:y_center+100,x_center-100:x_center+100]
data, new_wave = chop_data_cube(region, wave, na_rest-80, na_rest+80)


if os.path.exists("DATA/"+SN_name+"/"+"errcube.npy"):
    errcube = np.load("DATA/"+SN_name+"/"+"errcube.npy")
else:
    errcube = estimate_flux_error(data,new_wave,na_rest,kernel_size=100)
    np.save("DATA/"+SN_name+"/"+"errcube.npy",errcube)

#errcube = estimate_flux_error(data,new_wave,na_rest,kernel_size=100)
#np.save("DATA/"+SN_name+"/"+"errcube.npy",errcube)


errcube=np.transpose(errcube, (2, 0, 1)) #this can be optimized


i=findWavelengths(new_wave, na_rest)[1]

ny, nx = cube[i].shape

# coordinates of all pixels
x = np.arange(nx)
y = np.arange(ny)
xx, yy = np.meshgrid(x, y)

# flatten to 1D
x_flat = xx.ravel()
y_flat = yy.ravel()
signal = data[i].ravel()
noise  = errcube[i].ravel()

xy = np.column_stack([x_flat, y_flat])

target_sn = 50

additive = False

if additive:
    # 1. Additive case: Provide a pre-calculated array of pixel capacities.
    # This is efficient for capacities like (S/N)^2 with Poissonian noise.
    capacity_spec = (signal / noise)**2

else:
    # 2. Non-additive case: Provide a function for custom capacity logic.
    def capacity_spec(index):
        """Calculates (S/N)^2 for a bin from its pixel indices."""
        # Standard S/N formula for uncorrelated noise
        sn = np.sum(signal[index]) / np.sqrt(np.sum(noise[index]**2))
        # Example for correlated noise (see full example file for details):
        # sn /= 1 + 1.07 * np.log10(len(index))
        return sn**2

# Perform the binning. The target is target_sn**2 to match the capacity definition.
pow = PowerBin(xy, capacity_spec, target_capacity=target_sn**2)

# Plot the results. We use capacity_scale='sqrt' to display S/N instead of (S/N)^2.
pow.plot(capacity_scale='sqrt', ylabel='S/N')

plt.savefig("DATA/"+SN_name+"/"+"NEWVoronoi_bins.pdf", bbox_inches='tight')


plt.close()

