
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os, importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import functions
importlib.reload(functions)
from functions import *

# for gaussian fits
import sympy as sp
from IPython.display import display, Math, HTML
from astropy.table import Table
import sympy as sp

# CLI flags: --force recomputes caches, --resume explicitly reuses caches
import argparse
parser = argparse.ArgumentParser(description='Milky-Way processing')
parser.add_argument('--force', action='store_true', help='Recompute and overwrite cached .npy files')
parser.add_argument('--resume', action='store_true', help='Reuse existing cached .npy files when available (default)')
parser.add_argument('--save-temp', action='store_true', help='Write temporary npy files (into /auxiliar-results)')
args = parser.parse_args()
if args.force and args.resume:
    parser.error('Cannot use both --force and --resume')


SN_name="SN2011jm"#"SN2007cq"

#"SN2010ev"


#done "SN2019ehk" "SN2007cq" "SN2011jm"
#"SN2010ev" need to confirm if this stays the same after removing halpha stars

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
elif SN_name=="SN2019ehk":
    z=0.0043
    ra,dec=185.733958, 15.826119
elif SN_name=="SN2011jm":
    z=0.00326
    ra,dec=193.7129, 0.6541
elif SN_name=="SN2001el":
    z=0.003896
    ra,dec=56.12738,-44.63992

# Minimal: assume local DATA root ../../DATA
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DATA'))
data_dir_sn = os.path.join(DATA_ROOT, SN_name)
os.makedirs(data_dir_sn, exist_ok=True)
results_dir = os.path.join(data_dir_sn, 'auxiliar-results')
os.makedirs(results_dir, exist_ok=True)
file_name = os.path.join(data_dir_sn, SN_name + '.fits')


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
wave = np.array(CRVAL + CDELT * (np.arange(NAXIS) + 1 - CRPIX))

na_rest=(5890+5896)/2
index=findWavelengths(wave, na_rest)[1]

### defining a region for voronoi binning and Isophotes ###
y_center=int(y_len/2)
x_center=int(x_len/2)

width=70
region=cube[:,y_center-width:y_center+width,x_center-width:x_center+width]
region_chopped_Na, new_wave = chop_data_cube(region, wave, na_rest-100, na_rest+100)

## skipping this cube if its zero at Na
idx_lo = findWavelengths(wave, na_rest - 10)[1]
idx_hi = findWavelengths(wave, na_rest + 10)[1]
na_region_spectrum = region[idx_lo:idx_hi, :, :]

if np.all(np.isnan(na_region_spectrum)):
    print("Spectrum around Na D is all zeros in this region: skipping this cube.")
    sys.exit()

###

# import cube of uncertainties
if (not args.force) and os.path.exists(os.path.join(data_dir_sn, 'errcube.npy')):
    errcube = np.load(os.path.join(data_dir_sn, 'errcube.npy'))
    print("Read the error cube from a npy file")
    print(np.shape(errcube)) #this has to be the same (the uncertainty cube is computted for the region of interest only)
    print(np.shape(region[0]))
else:
    errcube = estimate_flux_error(region_chopped_Na,new_wave,na_rest,kernel_size=100)
    if args.resume and os.path.exists(os.path.join(data_dir_sn, 'errcube.npy')):
        # resume requested but file missing due to --force earlier; continue
        pass
    np.save(os.path.join(data_dir_sn, 'errcube.npy'),errcube)


##
"""stacked_cube=np.nanmedian(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##i was doing sum
#stacked_cube=np.nansum(cube[index-100:index+100, :, :], axis=0)
#stacked_cube = stacked_cube.astype(np.float32)


i=index

image = stacked_cube

if image.ndim == 3:
    image = np.mean(image, axis=-1)

mean, median, std = sigma_clipped_stats(image, sigma=3.0)

# using DAOStarFinder to detect stars
daofind = DAOStarFinder(fwhm=4.0, threshold=7*std)#5, 4
sources = daofind(image - median)"""

### new: only keeping sources that are not in halpha host emission

from scipy.spatial import cKDTree

# --- 1. Your existing continuum detection (unchanged) ---
stacked_cube = np.nanmedian(cube[index-100:index+100, :, :], axis=0)
image = stacked_cube
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
daofind = DAOStarFinder(fwhm=4.0, threshold=7*std)
sources = daofind(image - median)
if sources is not None:
    x_coords, y_coords = sources['x_centroid'], sources['y_centroid']

    # --- 2. Narrow-band Halpha image at the HOST's redshift ---
    ha_rest = 6563
    ha_obs = ha_rest * (1 + z)
    ha_index = findWavelengths(wave, ha_obs)[1]

    halfwidth = 10  # pixels, narrow -- just the line core
    halpha_image = np.nansum(cube[ha_index-halfwidth:ha_index+halfwidth, :, :], axis=0)

    mean_ha, median_ha, std_ha = sigma_clipped_stats(halpha_image, sigma=3.0)
    daofind_ha = DAOStarFinder(fwhm=4.0, threshold=5*std_ha)
    sources_ha = daofind_ha(halpha_image - median_ha)

    # --- 3. Cross-match: keep continuum sources with NO nearby Halpha source ---
    match_radius = 3.0

    if sources_ha is not None and len(sources_ha) > 0:
        ha_coords = np.transpose((sources_ha['x_centroid'], sources_ha['y_centroid']))
        cont_coords = np.transpose((x_coords, y_coords))

        tree = cKDTree(ha_coords)
        dist, _ = tree.query(cont_coords, k=1)

        is_mw_star = dist > match_radius
    else:
        is_mw_star = np.ones(len(sources), dtype=bool)

    sources_filtered = sources[is_mw_star]
    print(f"Kept {is_mw_star.sum()} / {len(sources)} sources after Halpha cross-match")




    sources=sources_filtered










    x_coords, y_coords = sources['x_centroid'], sources['y_centroid']
    print("We have detected ", len(sources)," sources!")
else:
    x_coords, y_coords = np.array([]), np.array([])
    print("No sources detected!")


##this is just needed for sn2010ev
ny, nx = (image).shape
if SN_name=="SN2010ev":
    
    x0, y0 = nx/2, ny/2
    d = np.hypot(x_coords - x0, y_coords - y0)
    #remove_idx = np.argmin(d)

    print("WARNING INSPECT THIS EXCLUSION")
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
#np.save(os.path.join(results_dir, 'masked_cube.npy'), masked_cube)
np.save(os.path.join(results_dir, 'mask.npy'), mask)
mw_mask=mask[y_center-width:y_center+width,x_center-width:x_center+width]



n_valid_pixels = np.count_nonzero(mask)

print("\nOriginal image had ", ny*nx," pixels, the one after masking MW stars has ", n_valid_pixels)


lo,up = np.nanpercentile(image,2),np.nanpercentile(image,98)
plt.contour(mask, levels=[0.5], colors='red', linewidths=1, origin='lower')
plt.imshow(image,cmap='Blues_r',origin='lower',clim=(lo,up))
plt.savefig(os.path.join(data_dir_sn, 'MW-masked-cube.pdf'), bbox_inches='tight')
plt.close()

## background plot

"""spec = np.nansum(cube[:,50:100,0:25], axis=(1, 2))
EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,save=os.path.join(data_dir_sn, 'background.pdf'))"""


## one single Av of median spectra using all spaxels, excluding MW stars
#skipping this
"""print("\nComputing sum spectra of all spaxels, excluding MW stars")

if (not args.force) and os.path.exists(os.path.join(results_dir, 'whole_masked_cube_spec.npy')):
    spec = np.load(os.path.join(results_dir, 'whole_masked_cube_spec.npy'))
    print("Read the masked cube from a npy file")
else:
    spec = np.nansum(masked_cube, axis=(1, 2))
    np.save(os.path.join(results_dir, 'whole_masked_cube_spec.npy'), spec)
    

out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,save=os.path.join(results_dir, 'MW-single-line-measurement.pdf'))
EW_all,ERR_all=out[0][0],out[1][0]"""


##

# random subset of spaxels, excluding MW stars
"""subset_cube, coords = random_spaxel_subset(masked_cube, mask, n_spaxels=500)
spec = np.nansum(subset_cube, axis=1)
out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=400,plots=False,KS=100,save=os.path.join(data_dir_sn,'MW-subset-line-measurement.pdf'))

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
plt.savefig(os.path.join(results_dir,'MW-diff-subsets-line-measurement.pdf'), bbox_inches='tight')
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

    np.save(os.path.join(results_dir, 'weighted_EWs.npy'), weighted_EWs)
    np.save(os.path.join(results_dir, 'weighted_EW_errs.npy'), weighted_EW_errs)

#

plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1,label="EW subsets of pixels")
plt.axhline(y=EW_all,label="EW using all pixels")
plt.axhspan(EW_all - ERR_all, EW_all + ERR_all,alpha=0.1)
plt.xlabel("Sizes S",fontsize=15)
plt.ylabel("Weighted EW from 50 random subsets of size S",fontsize=10)
plt.legend()
plt.savefig(os.path.join(results_dir, 'MW-inspecting-subset-sizes.pdf'), bbox_inches='tight')
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

plt.savefig(os.path.join(results_dir, 'Kron-ellipse.pdf'), bbox_inches='tight')
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
out=EW_voronoi_bins(np.array([spectrum]),wave,na_rest,v=400,plots=False,KS=100,save=os.path.join(results_dir,'Kron-ellipse-spectrum.pdf'))
EW_ellipse,ERR_ellipse=out[0][0],out[1][0]


# inspect best kernel size for continuum
continuum_bound_w_min=new_wave[0]
continuum_bound_w_max=new_wave[-1]
print(type(continuum_bound_w_min), continuum_bound_w_min)
print(type(continuum_bound_w_max), continuum_bound_w_max)

best_KS = best_continuum(wave, spectrum, na_rest,400,continuum_bound_w_min,continuum_bound_w_max,plots=True,save=os.path.join(results_dir,'Best-continuum.pdf'))

# inspect best window
best_window = best_integration_window(wave, spectrum, wavelength=na_rest,best_KS=best_KS,plots=True,save=os.path.join(results_dir,'Best-window.pdf'))

## Isophotes
if isophotes==True:
    # create expected output subfolders so saves won't fail
    for _d in ('isophotes-spec-sum', 'isophotes-spec-median'):
        os.makedirs(os.path.join(results_dir, _d), exist_ok=True)


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

    plt.savefig(os.path.join(results_dir,'isophotes.pdf'), bbox_inches='tight')

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


    plt.savefig(os.path.join(data_dir_sn,'isophotes.pdf'), bbox_inches='tight')

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
                if args.save_temp:
                    np.save(os.path.join(results_dir,'temp_window_cont.npy'), np.column_stack((wave, spec)))

            out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save=os.path.join(results_dir,'isophotes-spec-sum',str(int(sma))+'.pdf'))
            EWs_sum.append(out[0][0])
            EW_errs_sum.append(out[1][0])

        #median
        pix = masked_cube[:, mask]
        spec = np.nanmedian(pix, axis=1)
        
        if np.any(np.isnan(spec[index-100:index+100])):
            print("Spectrum is empty / all NaNs, skipping")
        else:
            out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save=os.path.join(results_dir,'isophotes-spec-median',str(int(sma))+'.pdf'))
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

    plt.savefig(os.path.join(results_dir,'isophotes_EWs_sum.pdf'), bbox_inches='tight')
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

    plt.savefig(os.path.join(results_dir,'isophotes_EWs_median.pdf'), bbox_inches='tight')
    plt.close()

    #final_EW_median_isophotes = EWs_median[np.argmax(np.divide(EWs_median, EW_errs_median))]
    #final_EW_median_isophotes_err = EW_errs_median[np.argmax(np.divide(EWs_median, EW_errs_median))]




####

"""

## EWs of random subsets of pixels ##
    if (not args.force) and os.path.exists(os.path.join(results_dir,'weighted_EWs.npy')):
    weighted_EWs = np.load(os.path.join(results_dir,'weighted_EWs.npy'))
    weighted_EW_errs = np.load(os.path.join(results_dir,'weighted_EW_errs.npy'))

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
                out=EW_voronoi_bins(np.array([spec]),wave,na_rest,v=best_window,plots=False,KS=best_KS,text=False,save=os.path.join(results_dir,'example-spectrum.pdf'))
            
        yy,ybar=weighted_average(EWs,EW_errs)
        weighted_EWs.append(yy)
        weighted_EW_errs.append(ybar)
    np.save(os.path.join(results_dir,'weighted_EWs.npy'), weighted_EWs)
    np.save(os.path.join(results_dir,'weighted_EW_errs.npy'), weighted_EW_errs)


plt.errorbar(sizes, weighted_EWs, yerr=weighted_EW_errs, fmt='o', c='Blue', capsize=5,zorder=1,label="EW subsets of pixels")
plt.axhline(y=EW_all,label="EW using all pixels")
plt.axhspan(EW_all - ERR_all, EW_all + ERR_all,alpha=0.1)

plt.axhline(y=EW_ellipse,label="EW an ellipse", color="Green")
plt.axhspan(EW_ellipse - ERR_ellipse, EW_ellipse + ERR_ellipse,alpha=0.1, color="Green")

plt.xlabel("Sizes S",fontsize=15)
plt.ylabel("Weighted EW from 50 random subsets of size S",fontsize=10)
plt.legend()
plt.savefig(os.path.join(results_dir,'All-MW-EW measurements.pdf'), bbox_inches='tight')
plt.close()

"""


## plot all values together here ##
#valor de EW dentro da isophote com maior SNR (com a mediana, com a media) são hlines
#valor EW obtido dos voronoi bins

## Voronoi binning

import voronoi2

centroids_vor, EWs_vor, EW_errs_vor, spectra_and_bins = voronoi2.binning(cube,new_wave,region_chopped_Na,errcube,wave,file_name,SN_name,z,na_rest,width,mw_mask,best_window,best_KS, results_dir=data_dir_sn,save_temp=args.save_temp)#,target_sn = target_snr)#750 for f/c#250

"""plt.scatter(centroids_vor, EWs_vor, c=np.divide(EWs_vor,EW_errs_vor),s=50, edgecolors='black', alpha=1,zorder=2)

if isophotes==True:
    plt.axhline(y=final_EW_sum_isophotes,label="EW from isophote (sum)")
    #plt.axhline(y=final_EW_median_isophotes,label="EW from isophote (median)")
    #plt.fill_between(x=centroids_vor,y1= final_EW_median_isophotes - final_EW_median_isophotes_err,y2= final_EW_median_isophotes + final_EW_median_isophotes_err,color='red',alpha=0.2)
    #plt.fill_between(x=centroids_vor,y1= final_EW_sum_isophotes - final_EW_sum_isophotes_err, y2= final_EW_sum_isophotes + final_EW_sum_isophotes_err,color='red',alpha=0.2)
plt.savefig(os.path.join(data_dir_sn,'All-EW-values.pdf'), bbox_inches='tight')
plt.close()"""



## Gaussian fits to the Na i D line

normalization=True####False##########################################True
voronoi=True

#import data
spec_voronoi_bins, bin_map, EWs_map_bins, binned_img, valid_bin_indices = spectra_and_bins
        


MODELS = {
    # Re-map to only run two models: A -> original H, B -> original J
    "A": (modelH, [-200, -150]),  # was H
    "B": (modelJ, [-200, -150, 10]),  # was J
}

results = {name: {"bic": [], "chi2r": [], "params": [], "models": []} for name in MODELS}

mask_wave     = (new_wave > na_rest - 18) & (new_wave < na_rest + 18)
wave_fit = new_wave[mask_wave]
continuum_bins = []

# finding the continuum of each bin
for i in range(len(spec_voronoi_bins)):
    #print("bin ",i)

    x,y=new_wave,spec_voronoi_bins[i]

    x_cont,y_cont=filterout_peaks(x,y,low=30, high=65,mode="both")#this used to be 30 and 70 respectively, but i found 20 and 65 is better

    # continuum
    delta_x = np.average(np.diff(x_cont))  # spacing in Å between x points
        
    kernel_size=best_KS
    kernel = cosine_kernel(kernel_size)
    cont = convolve1d(y_cont, kernel, mode='nearest')
    interp=interp1d(x_cont, cont, kind='cubic',fill_value="extrapolate")
        
    continuum=interp(new_wave)
    continuum_bins.append(continuum)

    ## subtracting the continuum to the bin spectra
    flux     = spec_voronoi_bins[i] - continuum
    flux_fit = flux[mask_wave]
    n=0
    
    for name, (model_fn, p0) in MODELS.items():
        #n+=1
        #print("N", n)
        try:
            popt, _ = curve_fit(
                model_fn, wave_fit, flux_fit,
                p0=p0,
                #bounds=MODEL_BOUNDS[name],
                maxfev=20000
            )
            mflux = model_fn(wave_fit, *popt)
            results[name]["bic"].append(compute_bic(flux_fit, mflux, len(popt)))
            results[name]["chi2r"].append(compute_reduced_chi2(flux_fit, mflux, len(popt)))
            results[name]["params"].append(popt)
            results[name]["models"].append(mflux)
        except RuntimeError:
            results[name]["bic"].append(np.nan)
            results[name]["chi2r"].append(np.nan)
            results[name]["params"].append(None)
            results[name]["models"].append(None)

rows = []
for name in MODELS:
    model_fn, _ = MODELS[name]
    n_params   = model_fn.__code__.co_argcount - 1
    bics       = np.array(results[name]["bic"],   dtype=float)
    chi2s      = np.array(results[name]["chi2r"], dtype=float)
    bic_median = np.nanmedian(bics)
    bic_mad    = np.nanmean(np.abs(bics - bic_median))
    chi2_med   = np.nanmedian(chi2s)
    chi2_mad   = np.nanmean(np.abs(chi2s - chi2_med))
    n_failed   = int(np.sum(np.isnan(bics)))
    rows.append((name, n_params, f"{bic_median:.1f} ± {bic_mad:.1f}",
                 f"{chi2_med:.3f} ± {chi2_mad:.3f}", n_failed))

table = Table(rows=rows,
              names=["Model", "N_params", "BIC (median ± MAD)", "χ²ᵣ (median ± MAD)", "N_failed"])
table.pprint(max_width=-1)


x = len(spec_voronoi_bins)


for model_name, (model_fn, _) in MODELS.items():
    model_params = results[model_name]["params"]
    model_fluxes = results[model_name]["models"]
    EWs_corr = np.full(x, np.nan)
    EWs_ori = np.full(x, np.nan)
    ncols = int(np.ceil(np.sqrt(x)))
    nrows = int(np.ceil(x / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(35, 25), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for bin_id in range(x):
        ax = axes[bin_id]
        if model_params[bin_id] is None:
            ax.set_title(f"Bin {bin_id} [FAILED]", fontsize=14, color="red")
            continue
        popt      = model_params[bin_id]
        mflux     = model_fn(wave_fit, *popt)
        data      = spec_voronoi_bins[bin_id] - continuum_bins[bin_id]
        continuum = continuum_bins[bin_id][mask_wave]

        if model_name == "A":
            emission_model   = np.zeros_like(wave_fit)
            absorption_model = mflux.copy()
        elif model_name == "B":
            # params: A1, A2, A3 — all sigmas fixed, all mu fixed
            A3, mu3, sigma3  = popt[-1], 5893, SIGMA_FIXED
            emission_model   = gaussian(wave_fit, abs(A3), mu3, sigma3)
            absorption_model = mflux - emission_model

        label = bin_id == 0
        ax.plot(new_wave, data, color="black", linewidth=2, alpha=0.5,
            label="Data" if label else None)
        ax.plot(wave_fit, mflux, color="red", linewidth=2, linestyle="dashed",
            label="Model" if label else None)
        if model_name == "B":
            ax.plot(wave_fit, absorption_model, color="blue", linewidth=2,
                linestyle="dashed",
                label="Model w/out emission" if label else None)
        bic_val  = results[model_name]["bic"][bin_id]
        chi2_val = results[model_name]["chi2r"][bin_id]
        ax.set_title(f"Bin {bin_id} ", fontsize=14)### |  BIC={bic_val:.1f}  χ²ᵣ={chi2_val:.2f}", fontsize=14)
        if label:
            ax.legend(fontsize=12)
        ax.set_xlim(na_rest - 30, na_rest + 30)
        ax.set_ylim(np.min(data) * 1.05, np.max(data) * 1.05)
        ax.tick_params(axis="both", labelsize=14)
        EWs_corr[bin_id] = -np.trapezoid(absorption_model / continuum, wave_fit) #absorption_model if flux-continuum
        EWs_ori[bin_id] = -np.trapezoid(data[mask_wave] / continuum, wave_fit) #data[mask_wave] is flux-continuum

    for ax in axes[x:]:
        ax.axis("off")
    safe_name = model_name.replace(" ", "_").replace("—", "-")
    fig.suptitle(model_name, fontsize=24, y=1.01)
    fig.supxlabel("Wavelength", fontsize=20)
    fig.supylabel("Flux", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir_sn,f"BIN_fits_{safe_name}.pdf"), bbox_inches='tight')
    plt.show()
    print(f"Saved: BIN_fits_{safe_name}.pdf")
    results[model_name]["EWs_corr"] = EWs_corr
    results[model_name]["EWs_ori"]  = EWs_ori


# Select the best-fitting model independently for each Voronoi bin spectrum.

best_model_by_bin = np.full(x, None, dtype=object)
for bin_id in range(x):
    bic_by_model = {
        model_name: results[model_name]["bic"][bin_id]
        for model_name in MODELS
    }
    valid_models = {
        name: bic for name, bic in bic_by_model.items()
        if np.isfinite(bic)
    }
    if valid_models:
        best_model_by_bin[bin_id] = min(valid_models, key=valid_models.get)

print("Best model by Voronoi bin:")
for model_name in MODELS:
    print(f"  Model {model_name}: {np.sum(best_model_by_bin == model_name)} bins")



x_s, A1, A2, A3 = sp.symbols('x A_1 A_2 A_3')
mu1, mu2, mu3 = sp.symbols('mu_1 mu_2 mu_3')
sigma1, sigma2, sigma3 = sp.symbols('sigma_1 sigma_2 sigma_3')
sM = sp.Symbol('sigma_{MUSE}')
delta = sp.Symbol('delta')

def G(A, mu, sigma):
    return A * sp.exp(-((x_s - mu)**2) / (2 * sigma**2))

na1 = sp.Symbol('mu_{Na,1}')
na2 = sp.Symbol('mu_{Na,2}')

models_sym = {
    "A": G(A1, na1,  sM)    + G(A2, na2, sM),
    "B": G(A1, na1,  sM)    + G(A2, na2, sM)    + G(A3, na_rest, sM)
}

# free parameters and fixed constraints per model
param_notes = {
    
    "A": ("$A_1, A_2$",
          "$\\mu_1, \\mu_2$ fixed to Na doublet,  $\\sigma_1 = \\sigma_2 = \\sigma_\\mathrm{MUSE}$,  no emission"),
    "B": ("$A_1, A_2, A_3$",
          "$\\mu_1, \\mu_2$ fixed to Na doublet,  $\\sigma_1 = \\sigma_2 = \\sigma_3 = \\sigma_\\mathrm{MUSE}$,  $\\mu_3 = 5893$ Å")
}

for name in MODELS:
    expr = models_sym[name]
    n_params = MODELS[name][0].__code__.co_argcount - 1
    display(HTML(f"<h4 style='margin-bottom:2px'>Model {name} &nbsp;|&nbsp; {n_params} free parameters</h4>"))
    display(Math(r'f(x) = ' + sp.latex(expr)))
    free, fixed = param_notes[name]
    display(HTML(
        f"<p style='margin-left:20px; font-size:13px; color:#333'>"
        f"<b>Free:</b> {free} &nbsp;|&nbsp; <b>Fixed:</b> <span style='color:gray'>{fixed}</span>"
        f"</p>"
    ))

##### EW histogram using the best BIC model selected independently for each bin

colors = [plt.cm.Set1(i) for i in [0, 1, 2]]
colors = ['#E74C3C', '#3498DB', "#57469A"]  # red, blue, green




def sigma_clip(arr, nsigma=2):
    arr  = np.array(arr)
    med  = np.median(arr)
    std  = np.std(arr)
    mask = np.abs(arr - med) < nsigma * std
    n_clipped = np.sum(~mask)
    if n_clipped:
        print(f"  Clipped {n_clipped} outliers beyond {nsigma}σ")
    return arr[mask]

# pick the bins that have a valid best-fitting model
selected_bins = [
    bin_id for bin_id, model_name in enumerate(best_model_by_bin)
    if model_name is not None
]
# EWs_ori_c and EWs_corr_c are the original and corrected EW, using the best-fitting model for each bin
EWs_ori_c = np.array([
    results[best_model_by_bin[bin_id]]["EWs_ori"][bin_id]
    for bin_id in selected_bins
])
EWs_corr_c = np.array([
    results[best_model_by_bin[bin_id]]["EWs_corr"][bin_id]
    for bin_id in selected_bins
])

median_ori  = np.median(EWs_ori_c)
sigma_ori   = np.std(EWs_ori_c)
median_corr = np.median(EWs_corr_c)
sigma_corr  = np.std(EWs_corr_c)

plt.figure(figsize=(11, 6))
plt.title("Spaxels EW measurements using the best model per bin")

plt.hist(EWs_ori_c,  label="EW from original spectra",     alpha=0.5,color=colors[0])
plt.axvline(x=median_ori, color=colors[0])
plt.axvspan(median_ori - sigma_ori, median_ori + sigma_ori, alpha=0.15, color=colors[0])

plt.hist(EWs_corr_c, label="EW from model w/out emission", alpha=0.5,color=colors[2])#aqui
plt.axvline(x=median_corr, color=colors[2])
plt.axvspan(median_corr - sigma_corr, median_corr + sigma_corr, alpha=0.15, color=colors[2])

plt.legend(fontsize=13)
plt.text(0.02, 0.96, f"Original:   Median={median_ori:.2f} Å,  σ={sigma_ori:.2f} Å",
         ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=13)
plt.text(0.98, 0.10, f"Corrected:  Median={median_corr:.2f} Å,  σ={sigma_corr:.2f} Å",
         ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=13)
plt.xlabel("EW (Å)", fontsize=13)
plt.ylabel("N bins", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(data_dir_sn,"EW_histogram_best_model_per_bin.pdf"), bbox_inches='tight')

## PLOTTING FINAL PLOTS: EW vs distance to the center and EW vs BIC
distance_indices = [
    index for index in valid_bin_indices
    if best_model_by_bin[index] is not None
]
distance_values = centroids_vor[
    [valid_bin_indices.index(index) for index in distance_indices]
]
distance_positions = [valid_bin_indices.index(index) for index in distance_indices]
distance_models = np.array([
    best_model_by_bin[index] for index in distance_indices
])
distance_ews = np.array([results[best_model_by_bin[index]]["EWs_corr"][index]
    for index in distance_indices
])
distance_ew_errs = np.asarray(EW_errs_vor)[distance_positions]
distance_bics = np.array([
    results[best_model_by_bin[index]]["bic"][index]
    for index in distance_indices
])

weighted_distance_ew, weighted_distance_err = weighted_average(distance_ews, distance_ew_errs)

model_colors = {"A": colors[0], "B": colors[1]}
fig, (ax_distance, ax_bic) = plt.subplots(2, 1, figsize=(11, 10), sharey=True)
for model_name, model_color in model_colors.items():
    model_mask = distance_models == model_name
    ax_distance.errorbar(
        distance_values[model_mask], distance_ews[model_mask],
        yerr=distance_ew_errs[model_mask], fmt="o", color=model_color,
        markeredgecolor="black", capsize=5, zorder=2,
        label=f"Model {model_name}"
    )
    ax_bic.errorbar(
        distance_bics[model_mask], distance_ews[model_mask],
        yerr=distance_ew_errs[model_mask], fmt="o", color=model_color,
        markeredgecolor="black", capsize=5, zorder=2,
        label=f"Model {model_name}"
    )

ax_distance.axhline(weighted_distance_ew, color=colors[2], linestyle="--",label="Weighted average")
ax_distance.axhspan(weighted_distance_ew - weighted_distance_err,weighted_distance_ew + weighted_distance_err,color=colors[2], alpha=0.15)
ax_bic.axhline(weighted_distance_ew, color=colors[2], linestyle="--",label="Weighted average")
ax_bic.axhspan(weighted_distance_ew - weighted_distance_err,weighted_distance_ew + weighted_distance_err,color=colors[2], alpha=0.15)
ax_distance.text(0.02, 0.05,f"Weighted average = {weighted_distance_ew:.2f} +/- {weighted_distance_err:.2f} Å",ha="left", va="bottom", transform=ax_distance.transAxes, fontsize=13)
ax_distance.set_xlabel("Distance from image center (px)", fontsize=13)
ax_distance.set_ylabel("Corrected EW (Å)", fontsize=13)
ax_distance.set_title("Corrected EW versus distance from image center")
ax_distance.legend(loc="upper right", fontsize=13)
ax_bic.set_xlabel("BIC of selected fit", fontsize=13)
ax_bic.set_ylabel("Corrected EW (Å)", fontsize=13)
ax_bic.set_title("Corrected EW versus BIC")
ax_bic.legend(loc="upper right", fontsize=13)
fig.suptitle(f"{SN_name}: Corrected EW measurements", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(data_dir_sn, "EW_corr_vs_distance.pdf"), bbox_inches="tight")
plt.close(fig)