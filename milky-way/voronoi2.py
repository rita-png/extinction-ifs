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


#### ----- importing data

data = fits.open(file_name)
cube = data[1].data   # this is the cube, a (3681 x 341 x 604) matrix with fluxes at different 3681 wavelengths and 308 x 318 spatial pixels ("spaxels")
header = data[1].header # this has information on the data cube
ecube = data[2].data # this is the cube uncertainty (3681 x 341 x 604)


x_len=len(cube[0][0])
y_len=len(cube[0])
CRVAL = float(header["CRVAL3"])
NAXIS = int(header["NAXIS3"])
CDELT = float(header["CD3_3"])
CRPIX = float(header["CRPIX3"])
wave = np.array(CRVAL + CDELT * (np.arange(NAXIS) - CRPIX))



na_rest=(5890+5896)/2

index=findWavelengths(wave, na_rest)[1]


#### ----- masking stars
stacked_cube=np.nansum(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)

i=index

image = stacked_cube

if image.ndim == 3:
    image = np.mean(image, axis=-1)

mean, median, std = sigma_clipped_stats(image, sigma=3.0)

daofind = DAOStarFinder(fwhm=5.0, threshold=4*std)
sources = daofind(image - median)

x_coords, y_coords = sources['xcentroid'], sources['ycentroid']

##this is just needed for sn2010ev
ny, nx = (image).shape
x0, y0 = nx/2, ny/2
d = np.hypot(x_coords - x0, y_coords - y0)
remove_idx = np.argmin(d)

x_coords = np.delete(x_coords, remove_idx)
y_coords = np.delete(y_coords, remove_idx)

print("Warning! Removing centermost star (for sn2010ev). Undo this for other SNe")

##

list(zip(x_coords, y_coords))
sources=list(zip(x_coords, y_coords))

sources = [[int(x) for x in row] for row in sources]
star_coords=np.array(sources)

masked_cube,mask=create_star_mask(cube, star_coords, radius=10)
np.save("DATA/"+SN_name+"/masked_cube.npy", masked_cube)
np.save("DATA/"+SN_name+"/mask.npy", mask)


lo,up = np.nanpercentile(image,2),np.nanpercentile(image,98)
plt.contour(mask.astype(int), levels=[0.5], colors='red', linewidths=1.5)
plt.imshow(image,cmap='Blues_r',origin='lower',clim=(lo,up))
plt.savefig("DATA/"+SN_name+"/"+"newmask.pdf", bbox_inches='tight')
plt.close()



mw_mask=mask
## -----



## Voronoi binning ##
y_center=int(y_len/2)
x_center=int(x_len/2)

region=cube[:,y_center-70:y_center+70,x_center-70:x_center+70]#cube[:,y_center-100:y_center+100,x_center-100:x_center+100]
mw_mask=mw_mask[y_center-70:y_center+70,x_center-70:x_center+70]
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

ny, nx = data[i].shape
yy, xx = np.mgrid[0:ny, 0:nx]
x = xx.ravel()
y = yy.ravel()
xy = np.column_stack([x, y])


#print(xy)


signal = data[i].ravel()
noise  = errcube[i].ravel()


target_sn = 200

additive = False


print(xy.shape)
print(signal.shape)
print(noise.shape)



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
#pow.plot(capacity_scale='sqrt', ylabel='S/N')
#plt.savefig("DATA/"+SN_name+"/"+"NEWVoronoi_bins.pdf", bbox_inches='tight')
#plt.close()



## converting the output into a binned image


bin_index = pow.xybin

aux=pow.bin_num


bin_map = pow.bin_num.reshape(ny, nx)

img = data[i]
binned_img = np.zeros_like(img)
EWs_map_bins=np.zeros_like(img)

n_bins = len(pow.bin_capacity) #number of bins

n_wave = len(new_wave)

#spectra_per_bin = []#np.zeros((n_bins, n_wave))
#err_per_bin = np.zeros((n_bins, n_wave))
centroids = []

EWs = []
EW_errs = []
center_y = ny / 2
center_x = nx / 2

k=0
for b in range(n_bins):
    mask = (bin_map == b)

    #just 2d image of wavelength of the absorption line
    binned_img[mask] = np.median(img[mask])


    #binning the spectra ONLY in bins that do not include MW stars
    if np.any(~mw_mask & mask):
        k+=1
        continue
    
    bin_pixels = data[:, mask]
    bin_pixels_err = errcube[:, mask]

    """spectra_per_bin.append(np.nansum(bin_pixels, axis=1))
    #err_per_bin.append(np.nansum(bin_pixels_err, axis=1)) """
    spectra_of_bin=np.nansum(bin_pixels, axis=1)

    ys, xs = np.where(mask)
    distances = np.sqrt((xs - center_x)**2 + (ys - center_y)**2)
    centroids.append(np.average(distances))

    #computing EWs
    a,b,foo=EW_voronoi_bins(np.array([spectra_of_bin]), new_wave,na_rest,v=400,plots=True,text=False)
    EWs.append(a[0])
    EW_errs.append(b[0])

    EWs_map_bins[mask] = a


EWs=np.asarray(EWs)
EW_errs=np.asarray(EW_errs)

#spectra_per_bin=np.array(spectra_per_bin)

print("Excluded ",k, " bins for including fluxes of the MW stars")

###
#spectra per bin

"""EWs, EW_errs, foo = EW_voronoi_bins(spectra_per_bin, new_wave,na_rest,v=400,plots=True,text=False)#EW_voronoi_bins(spectra_per_bin, new_wave, err_per_bin,na_rest,v=400,plots=True)
"""

SNRs=np.divide(EWs,EW_errs)
y = np.array(EWs)
sigma = np.array(EW_errs)

w = 1 / sigma**2

weighted_mean = np.sum(w * y) / np.sum(w)
mean_unc = np.sqrt(1 / np.sum(w))


weighted_std_dev = np.sqrt(np.sum(w * (y - weighted_mean)**2) / np.sum(w))
#std_dev = np.sqrt(np.sum((y -  weighted_mean)**2) / len(y))

plt.figure(figsize=(22, 6))


scatter=plt.errorbar(centroids, EWs, yerr=EW_errs, alpha=0.75, fmt='o', c='Blue', capsize=5,zorder=1)
scatter=plt.scatter(centroids, EWs, c=SNRs,s=50, edgecolors='black', alpha=1,zorder=2)
cbar=plt.colorbar(scatter)
cbar.set_label('SNR', fontsize=20) 
cbar.ax.tick_params(labelsize=20)
plt.xlabel("Distance from image center (px)",fontsize=20)
plt.ylabel("EW",fontsize=20)
plt.title("EW for each Voronoi bin",fontsize=20)
plt.text(0.02, 0.96, f"EW={weighted_mean:.2f} +/- {weighted_std_dev:.2f} (weigthed mean +/- weighted std_dev)", ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)
plt.text(0.02, 0.90, f"Mean uncertainty = {mean_unc:.4f}", ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)

plt.axhline(y=weighted_mean)

plt.fill_between(
    x=np.array([0, len(centroids)]),   # set these to your x-range
    y1=weighted_mean - weighted_std_dev,
    y2=weighted_mean + weighted_std_dev,
    color='red',
    alpha=0.2,
    label='Mean ± Error'
)
plt.tick_params(axis='both', which='major', labelsize=15)


plt.savefig("DATA/"+SN_name+"/"+"EWs_bins.pdf", bbox_inches='tight')
plt.show()



## plotting the binned image
fig, ax = plt.subplots(1, 3, figsize=(30, 8))

####

image = img
lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
cmap = plt.cm.Blues_r.copy()
im1 = ax[0].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[0],orientation="horizontal")
ax[0].set_title("Original fluxes",fontsize=20)
ax[0].tick_params(axis='both', which='major', labelsize=20)
cbar.ax.tick_params(labelsize=20)
ax[0].contour(mw_mask.astype(int), levels=[0.5], colors='red', linewidths=1.5)


image = binned_img
lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
cmap = plt.cm.Blues_r.copy()
im1 = ax[1].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[1],orientation="horizontal")
ax[1].set_title("Voronoi bins, using PowerBin",fontsize=20)
cbar.ax.tick_params(labelsize=20)
ax[1].tick_params(axis='both', which='major', labelsize=20)
cbar.set_label("Median fluxe inside bin", fontsize=20)
ax[1].contour(mw_mask.astype(int), levels=[0.5], colors='red', linewidths=1.5)

image = EWs_map_bins
lo, up = np.nanpercentile(image, 4), np.nanpercentile(image, 96)
cmap = plt.cm.Blues_r.copy()
im1 = ax[2].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[2],orientation="horizontal")
ax[2].set_title("EW in each bin",fontsize=20)
cbar.ax.tick_params(labelsize=20)
ax[2].tick_params(axis='both', which='major', labelsize=20)
cbar.set_label("EW", fontsize=20)
ax[2].contour(mw_mask.astype(int), levels=[0.5], colors='red', linewidths=1.5)


plt.savefig("DATA/"+SN_name+"/"+"NEWVoronoi_bins.pdf", bbox_inches='tight')
plt.close()
