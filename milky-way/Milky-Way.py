

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
"""stacked_cube=np.nansum(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##why the following?


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




# one single Av of median spectra using all spaxels, excluding MW stars
print("\nComputing median spectra of all spaxels, excluding MW stars")



if os.path.exists("whole_masked_cube_spec.npy"):
    spec = np.load("whole_masked_cube_spec.npy")
else:
    spec = np.nansum(masked_cube, axis=(1, 2))
    np.save("whole_masked_cube_spec.npy", spec)



x_chopped,y_chopped=chop_data(wave,spec,na_rest-50,na_rest+50)
plt.figure(figsize=(8,6))
plt.plot(x_chopped,y_chopped)   
plt.axvline(x=na_rest)
plt.savefig("MW-single-line-measurement.pdf", bbox_inches='tight')
out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=500,plots=True,KS=100)
print("\nEW function is ", out[0])



# random subset of spaxels, excluding MW stars
subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
median_spec = np.nansum(subset_cube, axis=1)
x_chopped,y_chopped=chop_data(wave,median_spec,na_rest-50,na_rest+50)
plt.figure(figsize=(8,6))
plt.plot(x_chopped,y_chopped)   
plt.axvline(x=na_rest)
plt.savefig("MW-subset-line-measurement.pdf", bbox_inches='tight')
out=EW_voronoi_bins(np.array([median_spec]),wave,na_rest,v=500,plots=False,KS=100)
print("\nEW function is ", out[0])

"""
# random subset of spaxels, excluding MW stars, diff subsets
EWs=[]
for i in range(0,10):
    subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
    median_spec = np.nansum(subset_cube, axis=1)
    out=EW_voronoi_bins(np.array([median_spec]),wave,na_rest,v=500,plots=False,KS=100)
    EWs.append(out[0])



plt.plot(np.arange(0,len(EWs)),EWs)   
plt.axvline(x=na_rest)

plt.savefig("MW-subset-line-measurement.pdf", bbox_inches='tight')



"""

#subs

#subset_sizes=range(0,n_valid_pixels)[500,]
