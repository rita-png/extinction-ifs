
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



import sys, os, importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import functions
importlib.reload(functions)
from functions import *


SN_name="SN2010ev"#"SN2007cq"#"ASASSN-14ad"#

print("WARNING change the following to true if you want isophotes")
isophotes=False

#potassium is 7665
if SN_name=="SN2010ev":
    z=0.00921
    ra, dec = 156.370792, -39.830889
    #target_snr = 250
elif SN_name=="SN2007cq":
    z=0.026018
    ra, dec = 333.66965, 5.078526
elif SN_name=="SN2007bm":
    z=0.006298
    ra, dec= 171.26039,	-9.795445
elif SN_name=="ASASSN-14ad":
    ra, dev=190.04742,	18.061644
    z = 0.0264#0.026464
    #target_snr = 150
elif SN_name=="CSP13aao":
    ra,dec=89.626465,-63.560677
    z = 0.061486
elif SN_name=="CSP13abl":#not analysed yet
    ra,dec=99.530464,-75.72468
    z=	0.040006
elif SN_name=="CSP14aaq":
    ra,dec=93.45008,-67.920715
    z=0.036518
elif SN_name=="LSQ12ca":
    ra,dec=82.765114,	-19.801537
    z=0.098752

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

### defining a region for voronoi binning and Isophotes ###
y_center=int(y_len/2)
x_center=int(x_len/2)

print("WARNING REMOCE THE FOLLOWIGN LINES")#change to 70 again
width=70#150
region=cube[:,y_center-width:y_center+width,x_center-width:x_center+width]
region_chopped_Na, new_wave = chop_data_cube(region, wave, na_rest-100, na_rest+100)


# import cube of uncertainties

if os.path.exists("DATA/"+SN_name+"/"+"errcube.npy"):
    errcube = np.load("DATA/"+SN_name+"/"+"errcube.npy")
    print("Read the error cube from a npy file")
    print(np.shape(errcube)) #this has to be the same (the uncertainty cube is computted for the region of interest only)
    print(np.shape(region[0]),"\n^^^^^^^^^^^^^^")
else:
    errcube = estimate_flux_error(region_chopped_Na,new_wave,na_rest,kernel_size=100)
    np.save("DATA/"+SN_name+"/"+"errcube.npy",errcube)


##





stacked_cube=np.nanmedian(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##i was doing sum
#stacked_cube=np.nansum(cube[index-100:index+100, :, :], axis=0)
#stacked_cube = stacked_cube.astype(np.float32)


i=index

image = stacked_cube

if image.ndim == 3:
    image = np.mean(image, axis=-1)

mean, median, std = sigma_clipped_stats(image, sigma=3.0)

# using DAOStarFinder to detect stars
daofind = DAOStarFinder(fwhm=4.0, threshold=7*std)#5, 4
sources = daofind(image - median)


# filter by compactness
#sources = sources[sources['peak'] / sources['npix'] > 1]


x_coords, y_coords = sources['xcentroid'], sources['ycentroid']
print("We have detected ", len(sources)," sources!")

##this is just needed for sn2010ev
ny, nx = (image).shape
x0, y0 = nx/2, ny/2
d = np.hypot(x_coords - x0, y_coords - y0)
#remove_idx = np.argmin(d)
r_exclude = 10

#x_coords = np.delete(x_coords, remove_idx)
#y_coords = np.delete(y_coords, remove_idx)

keep = d > r_exclude

x_coords = x_coords[keep]
y_coords = y_coords[keep]



print("Warning! Removing ", np.sum(~keep)," stars from the mask in the center of the image. The mask has a total of ",np.sum(keep)," masked stars.")

##

list(zip(x_coords, y_coords))
sources=list(zip(x_coords, y_coords))

sources = [[int(x) for x in row] for row in sources]
star_coords=np.array(sources)


# the following aimed to find mw stars from gaia cross matching
"""
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



## running the script to get star coords from hosphot

"""import mwstars

x, y = mwstars.return_matched_MW_stars(file_name,SN_name,z,ra,dec)
star_coords = np.column_stack((x.data, y.data))
stacked_cube=np.nansum(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##why the following?

i=index

image = stacked_cube

if image.ndim == 3:
    image = np.mean(image, axis=-1)"""



## saving output of create_star_mask

masked_cube,mask=create_star_mask(cube, star_coords, radius=10)
np.save("DATA/"+SN_name+"/masked_cube.npy", masked_cube)
np.save("DATA/"+SN_name+"/mask.npy", mask)
mw_mask=mask[y_center-width:y_center+width,x_center-width:x_center+width]



n_valid_pixels = np.count_nonzero(mask)

print("\nOriginal image had ", ny*nx," pixels, the one after masking MW stars has ", n_valid_pixels)


lo,up = np.nanpercentile(image,2),np.nanpercentile(image,98)
plt.contour(mask, levels=[0.5], colors='red', linewidths=1, origin='lower')
plt.imshow(image,cmap='Blues_r',origin='lower',clim=(lo,up))
plt.savefig("DATA/"+SN_name+"/MW-masked-cube.pdf", bbox_inches='tight')
plt.close()

## background plot

spec = np.nansum(cube[:,50:100,0:25], axis=(1, 2))
EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,save="DATA/"+SN_name+"/background.pdf")


## one single Av of median spectra using all spaxels, excluding MW stars
print("\nComputing sum spectra of all spaxels, excluding MW stars")

if os.path.exists("DATA/"+SN_name+"/whole_masked_cube_spec.npy"):
    spec = np.load("DATA/"+SN_name+"/whole_masked_cube_spec.npy")
    print("Read the masked cube from a npy file")

else:
    spec = np.nansum(masked_cube, axis=(1, 2))
    np.save("DATA/"+SN_name+"/whole_masked_cube_spec.npy", spec)
    

out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,save="DATA/"+SN_name+"/MW-single-line-measurement.pdf")
EW_all,ERR_all=out[0][0],out[1][0]


##

# random subset of spaxels, excluding MW stars
"""subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
spec = np.nansum(subset_cube, axis=1)
out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,save="DATA/"+SN_name+"/MW-subset-line-measurement.pdf")

##

# random subset of spaxels, excluding MW stars, diff subsets
print("Measuring EW of different subsets of spaxels")
EWs=[]
EW_errs=[]
SNRs=[]
for i in range(0,100):
    subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
    spec = np.nansum(subset_cube, axis=1)
    out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,text=False)
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
plt.savefig("DATA/"+SN_name+"/MW-diff-subsets-line-measurement.pdf", bbox_inches='tight')
plt.close()




# random subset of spaxels, excluding MW stars, diff subsets of diff sizes

sizes=np.linspace(100,5000,10)
sizes = [ int(x) for x in sizes ]

#####
#if os.path.exists("DATA/"+SN_name+"/weighted_EWs.npy"):
#    weighted_EWs = np.load("DATA/"+SN_name+"/weighted_EWs.npy")
#    weighted_EW_errs = np.load("DATA/"+SN_name+"/weighted_EW_errs.npy")
#else:
    
    


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
        out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,text=False,save=False)
        EWs.append(out[0][0])
        EW_errs.append(out[1][0])
            


    yy,ybar=weighted_average(EWs,EW_errs)
    weighted_EWs.append(yy)
    weighted_EW_errs.append(ybar)

np.save("DATA/"+SN_name+"/weighted_EWs.npy", weighted_EWs)
np.save("DATA/"+SN_name+"/weighted_EW_errs.npy", weighted_EW_errs)

#

plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1,label="EW subsets of pixels")
plt.axhline(y=EW_all,label="EW using all pixels")
plt.axhspan(EW_all - ERR_all, EW_all + ERR_all,alpha=0.1)
plt.xlabel("Sizes S",fontsize=15)
plt.ylabel("Weighted EW from 50 random subsets of size S",fontsize=10)
plt.legend()
plt.savefig("DATA/"+SN_name+"/MW-inspecting-subset-sizes.pdf", bbox_inches='tight')
plt.close()"""
#####




### Kron's ellipse ###
data=np.nansum(region[index-100:index+100, :, :], axis=0)#cube[index,:,:]
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

plt.savefig("DATA/"+SN_name+"/Kron-ellipse.pdf", bbox_inches='tight')
plt.close()

ny, nx = data.shape
cy, cx = ny/2, nx/2
kron_ellipse = min(objects, key=lambda obj: (obj['x'] - cx)**2 + (obj['y'] - cy)**2)#max(objects, key=lambda obj: obj['a'])

x0, y0 = kron_ellipse['x'], kron_ellipse['y']
a, b, theta = kron_ellipse['a'], kron_ellipse['b'], kron_ellipse['theta']

print(f"Galaxy center: ({x0:.2f}, {y0:.2f}), a={a:.2f}, b={b:.2f}")

kron_factor=1 #2.5

kron_a, kron_b = a * kron_factor, b * kron_factor

ny, nx = data.shape
y, x = np.mgrid[0:ny, 0:nx]

x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

mask = (x_rot / kron_a)**2 + (y_rot / kron_b)**2 <= 1

masked_cube = np.where(mask, region, np.nan)
masked_err_cube = np.where(mask, errcube, np.nan)

spectrum = np.nansum(masked_cube, axis=(1, 2))
#spectrum_err = np.sqrt(np.nansum(masked_err_cube**2))

out=EW_voronoi_bins(np.array([spectrum]),wave,na_rest,v=400,plots=False,KS=100,save="DATA/"+SN_name+"/Kron-ellipse-spectrum.pdf")
EW_ellipse,ERR_ellipse=out[0][0],out[1][0]


# inspect best kernel size for continuum

best_KS = best_continuum(wave, spectrum, wavelength=na_rest,vel=400,plots=True,save="DATA/"+SN_name+"/Best-continuum.pdf")

# inspect best window
best_window = best_integration_window(wave, spectrum, wavelength=na_rest,best_KS=best_KS,plots=True,save="DATA/"+SN_name+"/Best-window.pdf")

## Isophotes
if isophotes==True:
    ny, nx = data_sub.shape
    x0, y0 = nx / 2, ny / 2

    def dist(obj):
        return np.hypot(obj['x'] - x0, obj['y'] - y0)

    closest = min(candidate_galaxies, key=dist)

    x_center = closest['x']
    y_center = closest['y']
    a        = closest['a']
    b        = closest['b']
    theta    = closest['theta']


    geometry = EllipseGeometry(x0=x_center, y0=y_center, sma=a, eps=1-b/a,
                            pa=theta)

    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                            geometry.sma * (1 - geometry.eps), geometry.pa)


    ellipse = IsoEllipse(data, geometry)

    isolist = ellipse.fit_image()

    model_image = build_ellipse_model(data.shape, isolist)

    residual = data - model_image

    # plotting isophotes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(14, 5), nrows=2, ncols=2)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)


    lo,up = np.nanpercentile(data,2),np.nanpercentile(data,98)
    ax1.imshow(data, origin='lower',clim=(lo,up))

    # plotting ellipse with matplotlib
    ell = Ellipse((geometry.x0, geometry.y0), geometry.sma, (geometry.sma * (1 - geometry.eps)), angle=geometry.pa*180/np.pi,
                edgecolor='white', facecolor='none', linewidth=1.5)

    ax1.add_patch(ell)

    ax2.imshow(data, origin='lower',clim=(lo,up))
    ax2.set_title('Data')

    available_smas = [iso.sma for iso in isolist]
    available_smas = np.array(available_smas, dtype=float)
    print("The smas are originally", available_smas)
    available_smas = available_smas[available_smas > 3] #excluding centermost isophots, as they usually have a sma=0

    #picking 30 evenly spaced semi-major axis values, snapping each one to the nearest actually available isophote
    target_values = np.linspace(available_smas[0], available_smas[-1], 30)
    smas = np.array([available_smas[np.argmin(np.abs(available_smas - t))] for t in target_values])
    smas = np.unique(smas)

    print("The smas are", smas)


    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax2.plot(x, y, color='white')

    lo,up = np.nanpercentile(model_image,2),np.nanpercentile(model_image,98)
    ax3.imshow(model_image, origin='lower',clim=(lo,up))
    ax3.set_title('Ellipse Model')

    lo,up = np.nanpercentile(residual,2),np.nanpercentile(residual,98)
    ax4.imshow(residual, origin='lower',clim=(lo,up))
    ax4.set_title('Residual')

    plt.savefig("DATA/"+SN_name+"/isophotes.pdf", bbox_inches='tight')

    plt.close()




    #the following is to just output the smas image
    """fig, ax = plt.subplots()

    ax.imshow(data, origin='lower', vmin=lo, vmax=up)
    ax.set_title('Isophotes fit')

    available_smas = np.array([iso.sma for iso in isolist], float)
    available_smas = available_smas[available_smas > 3]

    target = np.linspace(available_smas[0], available_smas[-1], 15)
    smas = np.unique([available_smas[np.argmin(np.abs(available_smas - t))] for t in target])

    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y = iso.sampled_coordinates()
        ax.plot(x, y, color='white')


    plt.savefig("DATA/"+SN_name+"/isophotes.pdf", bbox_inches='tight')

    plt.close()"""


    #compute spectra inside each isophote

    EWs_sum=[]
    EW_errs_sum=[]
    EWs_median=[]
    EW_errs_median=[]
    k=0
    for sma in smas:
        
        iso = isolist.get_closest(sma)

        aper=EllipticalAperture((iso.x0, iso.y0),iso.sma,iso.sma * (1 - iso.eps),theta=iso.pa)

        mask = aper.to_mask(method='exact').to_image(data.shape)
        mask = mask.astype(bool)
        #spec_ellipse = cube * mask

        #sum
        spec_ellipse = masked_cube * mask
        spec = np.nansum(spec_ellipse, axis=(1, 2))

        #spec_ellipse_err = masked_err_cube * mask
        
        

        if np.any(np.isnan(spec[index-100:index+100])):
            print("Spectrum is empty / all NaNs, skipping")
        else:
            k+=1
            if k==1:
                print("OUTPUTTING SPEC OF SMALLEST SMA")
                np.save("DATA/"+SN_name+"/temp_window_cont.npy", np.column_stack((wave, spec)))

            out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save="DATA/"+SN_name+"/isophotes-spec-sum/"+str(int(sma))+".pdf")
            EWs_sum.append(out[0][0])
            EW_errs_sum.append(out[1][0])

        #median
        pix = masked_cube[:, mask]
        spec = np.nanmedian(pix, axis=1)
        
        if np.any(np.isnan(spec[index-100:index+100])):
            print("Spectrum is empty / all NaNs, skipping")
        else:
            out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save="DATA/"+SN_name+"/isophotes-spec-median/"+str(int(sma))+".pdf")
            EWs_median.append(out[0][0])
            EW_errs_median.append(out[1][0])






    # plotting EWs sum
    y = np.array(EWs_sum)
    w = 1 / np.array(EW_errs_sum)**2
    weighted_mean = np.nansum(w * y) / np.nansum(w)
    weighted_std_dev = np.sqrt(np.nansum(w * (y - weighted_mean)**2) / np.nansum(w))

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].errorbar(smas,EWs_sum, yerr=EW_errs_sum, fmt='o', c='Blue', capsize=5,zorder=1)
    ax[0].set_xlabel("Semi major axis",fontsize=12)
    ax[0].set_ylabel("EW",fontsize=12)
    ax[0].set_ylim([0,1])
    ax[1].set_ylim([0,20])
    ax[0].text(0.02, 0.96, f"EW={weighted_mean:.2f} +/- {weighted_std_dev:.2f}", ha='left', va='top', transform=ax[0].transAxes,fontsize=20)
    ax[0].set_title("EW for spectra inside isophotes, using sum of spectra",fontsize=15)
    ax[1].scatter(smas,np.divide(EWs_sum,EW_errs_sum))
    ax[1].set_xlabel("Semi major axis",fontsize=12)
    ax[1].set_ylabel("SNR",fontsize=12)

    plt.savefig("DATA/"+SN_name+"/isophotes_EWs_sum.pdf", bbox_inches='tight')
    plt.close()

    final_EW_sum_isophotes = EWs_sum[np.argmax(np.divide(EWs_sum, EW_errs_sum))]
    final_EW_sum_isophotes_err = EW_errs_sum[np.argmax(np.divide(EWs_sum, EW_errs_sum))]

    # plotting EWs median
    y = np.array(EWs_median)
    w = 1 / np.array(EW_errs_median)**2
    weighted_mean = np.nansum(w * y) / np.nansum(w)
    weighted_std_dev = np.sqrt(np.nansum(w * (y - weighted_mean)**2) / np.nansum(w))

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].errorbar(smas,EWs_median, yerr=EW_errs_median, fmt='o', c='Blue', capsize=5,zorder=1)
    ax[0].set_xlabel("Semi major axis",fontsize=12)
    ax[0].set_ylabel("EW",fontsize=12)
    ax[0].set_ylim([0,1])
    ax[1].set_ylim([0,20])
    ax[0].text(0.02, 0.96, f"EW={weighted_mean:.2f} +/- {weighted_std_dev:.2f}", ha='left', va='top', transform=ax[0].transAxes,fontsize=20)
    ax[0].set_title("EW for spectra inside isophotes, using median of spectra",fontsize=15)
    ax[1].scatter(smas,np.divide(EWs_median,EW_errs_median))
    ax[1].set_xlabel("Semi major axis",fontsize=12)
    ax[1].set_ylabel("SNR",fontsize=12)

    plt.savefig("DATA/"+SN_name+"/isophotes_EWs_median.pdf", bbox_inches='tight')
    plt.close()

    #final_EW_median_isophotes = EWs_median[np.argmax(np.divide(EWs_median, EW_errs_median))]
    #final_EW_median_isophotes_err = EW_errs_median[np.argmax(np.divide(EWs_median, EW_errs_median))]




####

"""

## EWs of random subsets of pixels ##
if os.path.exists("DATA/"+SN_name+"/weighted_EWs.npy"):
    weighted_EWs = np.load("DATA/"+SN_name+"/weighted_EWs.npy")
    weighted_EW_errs = np.load("DATA/"+SN_name+"/weighted_EW_errs.npy")

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
            out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save=False)
            EWs.append(out[0][0])
            EW_errs.append(out[1][0])
            if size==3366 and i==1:
                out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save="DATA/"+SN_name+"/example-spectrum.pdf")
            
        yy,ybar=weighted_average(EWs,EW_errs)
        weighted_EWs.append(yy)
        weighted_EW_errs.append(ybar)
    np.save("DATA/"+SN_name+"/weighted_EWs.npy", weighted_EWs)
    np.save("DATA/"+SN_name+"/weighted_EW_errs.npy", weighted_EW_errs)


plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1,label="EW subsets of pixels")
plt.axhline(y=EW_all,label="EW using all pixels")
plt.axhspan(EW_all - ERR_all, EW_all + ERR_all,alpha=0.1)

plt.axhline(y=EW_ellipse,label="EW an ellipse", color="Green")
plt.axhspan(EW_ellipse - ERR_ellipse, EW_ellipse + ERR_ellipse,alpha=0.1, color="Green")

plt.xlabel("Sizes S",fontsize=15)
plt.ylabel("Weighted EW from 50 random subsets of size S",fontsize=10)
plt.legend()
plt.savefig("DATA/"+SN_name+"/All-MW-EW measurements.pdf", bbox_inches='tight')
plt.close()

"""


## plot all values together here ##
#valor de EW dentro da isophote com maior SNR (com a mediana, com a media) são hlines
#valor EW obtido dos voronoi bins

## Voronoi binning

import voronoi2

centroids_vor, EWs_vor, EW_errs_vor = voronoi2.binning(cube,new_wave,region_chopped_Na,errcube,wave,file_name,SN_name,z,na_rest,width,mw_mask,best_window,best_KS)#,target_sn = target_snr)#750 for f/c#250

plt.scatter(centroids_vor, EWs_vor, c=np.divide(EWs_vor,EW_errs_vor),s=50, edgecolors='black', alpha=1,zorder=2)


plt.axhline(y=final_EW_sum_isophotes,label="EW from isophote (sum)")
#plt.axhline(y=final_EW_median_isophotes,label="EW from isophote (median)")

plt.fill_between(x=centroids_vor,y1= final_EW_sum_isophotes - final_EW_sum_isophotes_err, y2= final_EW_sum_isophotes + final_EW_sum_isophotes_err,color='red',alpha=0.2)
#plt.fill_between(x=centroids_vor,y1= final_EW_median_isophotes - final_EW_median_isophotes_err,y2= final_EW_median_isophotes + final_EW_median_isophotes_err,color='red',alpha=0.2)
plt.savefig("DATA/"+SN_name+"/All-EW-values.pdf", bbox_inches='tight')
plt.close()
