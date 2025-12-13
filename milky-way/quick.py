
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



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



stacked_cube=np.nansum(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##why the following?


i=index

image = stacked_cube

if image.ndim == 3:
    image = np.mean(image, axis=-1)

mean, median, std = sigma_clipped_stats(image, sigma=3.0)


x_coords = []
y_coords = []
list(zip(x_coords, y_coords))
sources=list(zip(x_coords, y_coords))

sources = [[int(x) for x in row] for row in sources]
star_coords=np.array(sources)

masked_cube,mask=create_star_mask(cube, star_coords, radius=10)
np.save("DATA/"+SN_name+"/masked_cube.npy", masked_cube)
np.save("DATA/"+SN_name+"/mask.npy", mask)



data=cube[index]
n_valid_pixels = np.count_nonzero(mask)

ny, nx = data.shape

lo,up = np.nanpercentile(data,2),np.nanpercentile(data,98)
plt.contour(mask, levels=[0.5], colors='red', linewidths=1, origin='lower')
plt.imshow(data,cmap='Blues_r',origin='lower',clim=(lo,up))
plt.savefig("DATA/"+SN_name+"/MW-masked-cube.pdf", bbox_inches='tight')
plt.close()




## Voronoi binning

import voronoi2

centroids_vor, EWs_vor, EW_errs_vor = voronoi2.binning(cube,wave,file_name,SN_name,z,na_rest,mask,target_sn = 200)

plt.close()

