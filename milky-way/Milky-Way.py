
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



import sys, os, importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import functions
importlib.reload(functions)
from functions import *
z=0.00921

file_name="SN2010ev.fits"


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




#commented the following so that i only input stars_coords
"""

from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import Angle

stacked_cube=np.nansum(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##why the following?


i=index

image = stacked_cube

if image.ndim == 3:
    image = np.mean(image, axis=-1)

mean, median, std = sigma_clipped_stats(image, sigma=20.0)

# using DAOStarFinder to detect stars
daofind = DAOStarFinder(fwhm=20.0, threshold=1.0*std)
sources = daofind(image - median)

x_coords, y_coords = sources['xcentroid'], sources['ycentroid']



print("Found ", len(x_coords), " sources!\n")


# Checking whether the sources are in Gaia catalogue
wcs = WCS(data[1].header) 

center_x=int(x_len/2)
center_y=int(y_len/2)

ra, dec, _ = wcs.all_pix2world(center_x, center_y, 0, 0)

#print(ra,dec)




list(zip(x_coords, y_coords))
sources=list(zip(x_coords, y_coords))

sources = [[int(x) for x in row] for row in sources]
sources=np.array(sources)



star_ra,star_dec,star_par,star_parer=match_gaia(sources,header,ra,dec)




out=gaia_parameters(star_ra,star_dec)



ra_hms = Angle(ra, unit=u.deg).to_string(unit=u.hour, sep=':')


stars_data = pd.DataFrame({
    'x': sources[:,0],               # original image x
    'y': sources[:,1],               # original image y
    'ra': star_ra,                   # Right Ascension
    'dec': star_dec,                 # Declination
    'parallax': out[0],              # Parallax
    'parallax_err': out[1],          # Parallax error
    'teff': out[2],                  # Effective temperature
    'logg':out[3],                   # Surface gravity
    'met': out[4],                   # Metallicity
    'mag': out[5]                    # Mean magnitude in g-band
})


stars_data = stars_data.dropna()
stars_data


star_coords=np.array(stars_data[['x', 'y']].values)"""

star_coords=np.array([[154, 40], [174, 109], [200, 236],[237, 273]])


#saving output of create_star_mask

if os.path.exists("masked_cube.npy"):
    masked_cube = np.load("masked_cube.npy")
    mask = np.load("mask.npy")
else:
    masked_cube,mask=create_star_mask(cube, star_coords, radius=15)
    np.save("masked_cube.npy", masked_cube)
    np.save("mask.npy", mask)




data=cube[index]
n_valid_pixels = np.count_nonzero(mask)

ny, nx = data.shape
print("\nOriginal image had ", ny*nx," pixels, the one after masking MW stars has ", n_valid_pixels)


lo,up = np.nanpercentile(data,2),np.nanpercentile(data,98)
plt.contour(mask, levels=[0.5], colors='red', linewidths=1, origin='lower')
plt.imshow(data,cmap='Blues_r',origin='lower',clim=(lo,up))
plt.savefig("MW-masked-cube.pdf", bbox_inches='tight')
plt.close()




## one single Av of median spectra using all spaxels, excluding MW stars
print("\nComputing median spectra of all spaxels, excluding MW stars")

if os.path.exists("whole_masked_cube_spec.npy"):
    spec = np.load("whole_masked_cube_spec.npy")
else:
    spec = np.nansum(masked_cube, axis=(1, 2))
    np.save("whole_masked_cube_spec.npy", spec)

out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=500,plots=False,KS=100,save="MW-single-line-measurement.pdf")

##

# random subset of spaxels, excluding MW stars
subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
spec = np.nansum(subset_cube, axis=1)
out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=500,plots=False,KS=100,save="MW-subset-line-measurement.pdf")

##

# random subset of spaxels, excluding MW stars, diff subsets
print("Measuring EW of different subsets of spaxels")
EWs=[]
EW_errs=[]
SNRs=[]
for i in range(0,100):
    subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
    spec = np.nansum(subset_cube, axis=1)
    out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=500,plots=False,KS=100,text=False)
    EWs.append(out[0][0])
    EW_errs.append(out[1][0])
    


SNRs=np.divide(EWs,EW_errs)

scatter=plt.errorbar(np.arange(0,len(EWs)), EWs, yerr=EW_errs, fmt='o', c='Blue', capsize=5,zorder=1)
scatter=plt.scatter(np.arange(0,len(EWs)),EWs,c=SNRs,s=100,zorder=2)
plt.colorbar(scatter, label='SNR')
plt.xlabel("# Subset of pixels",fontsize=15)
plt.ylabel(" EW for a given subset of pixels",fontsize=15)
yy,ybar=weighted_average(EWs,EW_errs)
plt.axhline(y=yy)
plt.axhspan(yy - ybar, yy + ybar,alpha=0.1)
plt.savefig("MW-diff-subsets-line-measurement.pdf", bbox_inches='tight')
plt.close()




# random subset of spaxels, excluding MW stars, diff subsets of diff sizes

sizes=np.linspace(100,5000,10)
sizes = [ int(x) for x in sizes ]

print("Measuring EW of different subsets of spaxels of different sizes ",sizes)

if os.path.exists("weighted_EWs.npy"):
    weighted_EWs = np.load("weighted_EWs.npy")
    weighted_EW_errs = np.load("weighted_EW_errs.npy")
else:
    
    


    weighted_EWs=[]
    weighted_EW_errs=[]
    for size in sizes:
        print("\nsize ",size)
        EWs=[]
        EW_errs=[]
        SNRs=[]
        for i in range(0,50):
            subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=size)
            spec = np.nansum(subset_cube, axis=1)
            out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=500,plots=False,KS=100,text=False,save=False)
            EWs.append(out[0][0])
            EW_errs.append(out[1][0])
            


        yy,ybar=weighted_average(EWs,EW_errs)
        weighted_EWs.append(yy)
        weighted_EW_errs.append(ybar)

        np.save("weighted_EWs.npy", masked_cube)
        np.save("weighted_EW_errs.npy", mask)


plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1)
plt.xlabel("Sizes",fontsize=15)
plt.ylabel("Weighted EW ......",fontsize=15)
plt.close()

#does this depend on the size of the subset?



## Voronoi binning ##