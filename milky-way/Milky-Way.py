
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



# the following aimed to find mw stars from gaia cross matching
"""
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



lo,up = np.nanpercentile(image,2),np.nanpercentile(image,98)
plt.scatter(x_coords, y_coords, s=50, edgecolor='red', facecolor='none', label="Detected Sources")
plt.imshow(image,cmap='Blues_r',origin='lower',clim=(lo,up))
plt.savefig("all-detected-sources.pdf", bbox_inches='tight')
plt.close()


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


star_coords=np.array(stars_data[['x', 'y']].values)

star_coords=np.array([[154, 40], [174, 109], [200, 236],[237, 273]])
"""

## running the script to 
import mwstars
print(mwstars.x_matched)  # 10
print(mwstars.y_matched)  # 20
"""
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
EW_all,ERR_all=out[0][0],out[1][0]


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

print("(dommented NOT Measuring EW of different subsets of spaxels of different sizes ",sizes)

"""if os.path.exists("weighted_EWs.npy"):
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

    np.save("weighted_EWs.npy", weighted_EWs)
    np.save("weighted_EW_errs.npy", weighted_EW_errs)


plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1,label="EW subsets of pixels")
plt.axhline(y=EW_all,label="EW using all pixels")
plt.axhspan(EW_all - ERR_all, EW_all + ERR_all,alpha=0.1)
plt.xlabel("Sizes S",fontsize=15)
plt.ylabel("Weighted EW from 50 random subsets of size S",fontsize=10)
plt.legend()
plt.savefig("MW-inspecting-subset-sizes.pdf", bbox_inches='tight')
plt.close()
"""
#does this depend on the size of the subset?

## Kron's ellipse ##

data=np.nansum(cube[index-100:index+100, :, :], axis=0)#cube[index,:,:]
data = data.astype(np.float32)
bkg = sep.Background(data)
data_sub = data - bkg


objects = sep.extract(data_sub, thresh=7)

lo,up = np.nanpercentile(data_sub,2),np.nanpercentile(data_sub,98)
plt.imshow(data_sub, cmap='gray', origin='lower',clim=(lo,up))

plt.title("Candidate galaxies")

candidate_galaxies = [obj for obj in objects if obj['a'] > 5] 

for obj in candidate_galaxies:
    x_center, y_center, a, b, theta = obj['x'], obj['y'], obj['a'], obj['b'], obj['theta']
    ell = Ellipse((x_center, y_center), a, b, angle=np.degrees(theta), edgecolor='red', facecolor='none', alpha=0.8)
    plt.gca().add_patch(ell)
    
    print(a,b)

plt.savefig("Kron-ellipse.pdf", bbox_inches='tight')
plt.close()

ny, nx = data.shape
cy, cx = ny/2, nx/2
kron_ellipse = min(objects, key=lambda obj: (obj['x'] - cx)**2 + (obj['y'] - cy)**2)#max(objects, key=lambda obj: obj['a'])

x0, y0 = kron_ellipse['x'], kron_ellipse['y']
a, b, theta = kron_ellipse['a'], kron_ellipse['b'], kron_ellipse['theta']

print(f"Galaxy center: ({x0:.2f}, {y0:.2f}), a={a:.2f}, b={b:.2f}")

kron_factor=2.5

kron_a, kron_b = a * kron_factor, b * kron_factor

nz, ny, nx = cube.shape
y, x = np.mgrid[0:ny, 0:nx]

x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

mask = (x_rot / kron_a)**2 + (y_rot / kron_b)**2 <= 1

masked_cube = np.where(mask, cube, np.nan)


spectrum = np.nansum(masked_cube, axis=(1, 2))


out=EW_voronoi_bins(np.array([spectrum]),wave,na_rest,v=500,plots=False,KS=100,save="Kron-ellipse-spectrum.pdf")
EW_ellipse,ERR_ellipse=out[0][0],out[1][0]

## plot all values together here ##
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

    np.save("weighted_EWs.npy", weighted_EWs)
    np.save("weighted_EW_errs.npy", weighted_EW_errs)


plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1,label="EW subsets of pixels")
plt.axhline(y=EW_all,label="EW using all pixels")
plt.axhspan(EW_all - ERR_all, EW_all + ERR_all,alpha=0.1)

plt.axhline(y=EW_ellipse,label="EW an ellipse", color="Green")
plt.axhspan(EW_ellipse - ERR_ellipse, EW_ellipse + ERR_ellipse,alpha=0.1, color="Green")

plt.xlabel("Sizes S",fontsize=15)
plt.ylabel("Weighted EW from 50 random subsets of size S",fontsize=10)
plt.legend()
plt.savefig("All-MW-EW measurements.pdf", bbox_inches='tight')
plt.close()


## Voronoi binning ##
y_center=int(y_center)
x_center=int(x_center)
region=cube[:,y_center-100:y_center+100,x_center-100:x_center+100]
data, new_wave = chop_data_cube(region, wave, na_rest-80, na_rest+80)#could use cube or zoom-in in the center


if os.path.exists("errcube.npy"):
    errcube = np.load("errcube.npy")
else:
    errcube = estimate_flux_error(data,new_wave,na_rest,kernel_size=100)
    np.save("errcube.npy",errcube)


errcube=np.transpose(errcube, (2, 0, 1)) #this can be cleaned


i=findWavelengths(new_wave, na_rest)[1]
if os.path.exists("voronoi_bins.npy"):
    voronoi_bins = np.load("voronoi_bins.npy")
else:
    voronoi_bins=voronoi(data[i],errcube[i],target_snr=40,pixel_size=0.2,plots=False,text=True)
    np.save("voronoi_bins.npy",errcube)



fig, ax = plt.subplots(1, 2, figsize=(20, 8))

####

image = data[i]
lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
cmap = plt.cm.Blues_r.copy()
im1 = ax[0].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[0])#, orientation="horizontal")
ax[0].set_title("Original fluxes",fontsize=30)
ax[0].tick_params(axis='both', which='major', labelsize=30)
cbar.ax.tick_params(labelsize=30)



image = voronoi_bins[0]
lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
cmap = plt.cm.Blues_r.copy()
im1 = ax[1].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[1])#, orientation="horizontal")
ax[1].set_title("Voronoi bins",fontsize=30)
cbar.ax.tick_params(labelsize=30)
ax[1].tick_params(axis='both', which='major', labelsize=30)

plt.savefig("Voronoi_bins.pdf", bbox_inches='tight')
plt.close()

# EW per voronoi bin

spectra_per_bin,err_per_bin = apply_voronoi_to_cube(data,errcube,voronoi_bins[1])

EWs, EW_errs, SNRs = EW_voronoi_bins(spectra_per_bin, new_wave,na_rest,v=400,plots=True)#EW_voronoi_bins(spectra_per_bin, new_wave, err_per_bin,na_rest,v=400,plots=True)

y = np.array(EWs)
sigma = np.array(EW_errs)

w = 1 / sigma**2

weighted_mean = np.sum(w * y) / np.sum(w)
weighted_std_mean = np.sqrt(1 / np.sum(w))
weighted_spread = np.sqrt(np.sum(w * (y - weighted_mean)**2) / np.sum(w))

plt.figure(figsize=(14, 6))

x_pos = np.linspace(1, len(EWs),len(EWs))


scatter=plt.errorbar(x_pos, EWs, yerr=EW_errs, alpha=0.75, fmt='o', c='Blue', capsize=5,zorder=1)
scatter=plt.scatter(x_pos, EWs, c=SNRs,s=50, edgecolors='black', alpha=1,zorder=2)
cbar=plt.colorbar(scatter)
cbar.set_label('SNR', fontsize=30) 
cbar.ax.tick_params(labelsize=30)
plt.xlabel("Voronoi bin",fontsize=30)
plt.ylabel("EW",fontsize=30)
plt.title("EW for each Voronoi bin",fontsize=30)

plt.axhline(y=weighted_mean)

plt.fill_between(
    x=np.array([0, len(x_pos)]),   # set these to your x-range
    y1=weighted_mean - weighted_spread,
    y2=weighted_mean + weighted_spread,
    color='red',
    alpha=0.2,
    label='Mean ± Error'
)
plt.tick_params(axis='both', which='major', labelsize=30)


plt.savefig("EWs_bins.pdf", bbox_inches='tight')
plt.show()"""
