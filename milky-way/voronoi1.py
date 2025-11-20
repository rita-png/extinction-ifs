
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
if os.path.exists("DATA/"+SN_name+"/"+"voronoi_bins.npy"):
    voronoi_bins = np.load("DATA/"+SN_name+"/"+"voronoi_bins.npy")
else:
    voronoi_bins=voronoi(data[i],errcube[i],target_snr=60,pixel_size=0.2,plots=False,text=True)
    np.save("DATA/"+SN_name+"/"+"voronoi_bins.npy",errcube)



fig, ax = plt.subplots(1, 2, figsize=(20, 8))

####

image = data[i]
lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
cmap = plt.cm.Blues_r.copy()
im1 = ax[0].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[0])
ax[0].set_title("Original fluxes",fontsize=30)
ax[0].tick_params(axis='both', which='major', labelsize=30)
cbar.ax.tick_params(labelsize=30)



image = voronoi_bins[0]
lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
cmap = plt.cm.Blues_r.copy()
im1 = ax[1].imshow(image, cmap=cmap, origin='lower', clim=(lo, up))
cbar=fig.colorbar(im1, ax=ax[1])
ax[1].set_title("Voronoi bins",fontsize=30)
cbar.ax.tick_params(labelsize=30)
ax[1].tick_params(axis='both', which='major', labelsize=30)

plt.savefig("DATA/"+SN_name+"/"+"Voronoi_bins.pdf", bbox_inches='tight')
plt.close()

# EW per voronoi bin

#summing the spectra inside each bin
#print(np.unique(voronoi_bins, return_counts=True))

spectra_per_bin,err_per_bin = apply_voronoi_to_cube(data,errcube,voronoi_bins[1])
print(spectra_per_bin)
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
plt.text(0.02, 0.98, f"EW={weighted_mean:.2f} +/- {weighted_spread:.2f}", ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)

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


plt.savefig("DATA/"+SN_name+"/"+"EWs_bins.pdf", bbox_inches='tight')
plt.show()