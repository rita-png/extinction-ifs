
import sys, os, importlib




na_rest=(5890+5896)/2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import functions
importlib.reload(functions)
from functions import *


def return_matched_MW_stars(file_name,SN_name,z,ra,dec):

    #file_name="SN2010ev.fits"
    name = SN_name#'SN2010ev'
    #z=0.00921


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




    index=findWavelengths(wave, na_rest)[1]






    stacked_cube=np.nansum(cube[int(len(wave)/4):int(3*len(wave)/4),:,:], axis=0)##why the following?


    image = stacked_cube

    if image.ndim == 3:
        image = np.mean(image, axis=-1)

    mean, median, std = sigma_clipped_stats(image, sigma=20.0)



    


    #get MUSE images wcs 
    f2 = fits.open(file_name) #need to change to your MUSE .fits file (to get the astronomical coordinate system of the image)


    img_wcs2 = WCS(f2[1].header).celestial

    hdr = f2[1].header
    w = WCS(hdr).celestial
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    x_c, y_c = (nx - 1)/2, (ny - 1)/2
    center = w.pixel_to_world(x_c, y_c)
    #print("Center RA, Dec =", center.ra.deg, center.dec.deg)

    host_ra, host_dec = center.ra.deg, center.dec.deg


    survey = "SkyMapper"





    download_images(name, 

                    host_ra, 

                    host_dec, 

                    survey=survey, 

                    overwrite=False,  # saves some time when rerunning

                    save_input=True,  # to be able to replicate the results - True by default

                    )



    #get survey images wcs 



    f = fits.open(f'images/{name}/{survey}/{survey}_r.fits')

    img_wcs = WCS(f[0].header)




    coadd_images(name, filters='r', survey=survey)



    masking.create_mask(name, 

                        host_ra, 

                        host_dec, 

                        filt='r', 

                        survey=survey, # masking parameters

                        threshold=12,  # sigmas above bkg to detect sources <----

                        sigma=8,  # width of the Gaussian kernel <-------

                        r=4,  # to scale the size of the masks

                        # other parameters

                        ra=ra, dec=dec,  # to plot the SN position

                        save_plots=True,  # False by default as it takes some time to create the figure

                        save_mask_params=True,  # to use the output on the other images - False by default

                        save_input=True, 

                        crossmatch=True # option of the crossmatch with gaia <------

                    )




    coord_match=pd.read_csv(f"images/{name}/{survey}/mask_parameters_r.csv")



    #Get x and y coordinates of matched objects in pixels
    detected_x=coord_match["x"].to_numpy() #Returns the x and y coordinates of the matching objects

    detected_y=coord_match["y"].to_numpy()





    #convert pixels to ra and dec in degrees
    objs_coord = img_wcs.pixel_to_world(coord_match["x"], coord_match["y"]) 


    objs_ra = objs_coord.ra.deg

    objs_dec = objs_coord.dec.deg



    #return objects coordinates in pixel coordinates of MUSE image 

    muse_x, muse_y = img_wcs2.world_to_pixel(objs_coord)



    survey_hdu = fits.open(f'images/{name}/{survey}/{survey}_r.fits')[0]
    survey_img, survey_wcs = survey_hdu.data, WCS(survey_hdu.header)
    muse_hdu = fits.open(file_name)[1]
    muse_cube, muse_header = muse_hdu.data, muse_hdu.header
    muse_img = np.nansum(muse_cube[int(muse_cube.shape[0]/4):int(3*muse_cube.shape[0]/4)], axis=0)
    muse_wcs = WCS(muse_header).celestial

    # convert ra dec of objects to coords
    survey_x, survey_y = survey_wcs.world_to_pixel(objs_coord)
    muse_x, muse_y = muse_wcs.world_to_pixel(objs_coord)

    # using DAOstar finder to see sources in MUSE image


    mean, median, std = sigma_clipped_stats(muse_img, sigma=20.0)

    daofind = DAOStarFinder(fwhm=10.0, threshold=1*std)
    sources = daofind(image - median)

    x_coords, y_coords = sources['xcentroid'], sources['ycentroid']

    print("Found ", len(x_coords), " sources!\n")

    # keep x_coords, y_coords close to muse_x, muse_y

    tol = 15

    dx = x_coords[:, None] - muse_x[None, :]
    dy = y_coords[:, None] - muse_y[None, :]
    dist = np.sqrt(dx**2 + dy**2)

    min_dist = np.min(dist, axis=1)
    matched_idx = np.where(min_dist < tol)[0]


    x_matched = x_coords[matched_idx]
    y_matched = y_coords[matched_idx]


    ##this is just needed for sn2010ev
    ny, nx = (cube[100]).shape
    x0, y0 = nx/2, ny/2

    d = np.hypot(x_matched - x0, y_matched - y0)

    remove_idx = np.argmin(d)

    x_matched = np.delete(x_matched, remove_idx)
    y_matched = np.delete(y_matched, remove_idx)


    print(f"Matched {len(x_matched)} / {len(x_coords)} sources within {tol} px tolerance.")
    print("Warning! Removing centermost star (for sn2010ev). Undo this for other SNe")




    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # survey image with catalogue sources
    ax1 = fig.add_subplot(2, 2, 1, projection=survey_wcs)
    ax1.imshow(survey_img, origin='lower', cmap='Greys',
            norm=simple_norm(survey_img, 'sqrt', percent=99))
    ax1.scatter(survey_x, survey_y, s=50, edgecolor='red', facecolor='none')
    ax1.set_title(f"{survey} "+SN_name)
    ax1.set_xlabel('RA')
    ax1.set_ylabel('Dec')

    # MUSE image with catalogue sources
    ax2 = axes[0, 1]
    ax2.imshow(muse_img, origin='lower', cmap='Greys',
            norm=simple_norm(muse_img, 'sqrt', percent=99))
    ax2.scatter(muse_x, muse_y, s=50, edgecolor='red', facecolor='none')
    ax2.set_title("MUSE "+SN_name)
    ax2.set_xlabel('RA')
    ax2.set_ylabel('Dec')

    # DAOStar sources found
    lo, up = np.nanpercentile(image, 2), np.nanpercentile(image, 98)
    axes[1, 0].imshow(image, cmap='Blues_r', origin='lower', clim=(lo, up))
    axes[1, 0].scatter(x_coords, y_coords, s=50, edgecolor='red', facecolor='none')
    axes[1, 0].set_title("DAOStar detected")
    axes[1, 0].set_xlabel('RA')
    axes[1, 0].set_ylabel('Dec')

    #matches
    axes[1, 1].imshow(image, cmap='Blues_r', origin='lower', clim=(lo, up))
    axes[1, 1].scatter(x_matched, y_matched, s=50, edgecolor='red', facecolor='none')
    axes[1, 1].set_title("Matched sources")
    axes[1, 1].set_xlabel('RA')
    axes[1, 1].set_ylabel('Dec')

    plt.tight_layout()
    plt.savefig("DATA/"+SN_name+"/"+"all-detected-sources.pdf", bbox_inches='tight')
    plt.close()

    return x_matched,y_matched

