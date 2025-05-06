import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from scipy.ndimage import convolve1d
import math
from astropy.wcs import WCS
import matplotlib.animation as animation
import pandas as pd

from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

from species import SpeciesInit
from species.data.database import Database
from species.read.read_model import ReadModel
from species.plot.plot_spectrum import plot_spectrum

## visualization ##

def plot_image(image, wavelengths, index, colormap,title=''):
    if np.ndim(image)==2:
        lo,up = np.nanpercentile(image,2),np.nanpercentile(image,98)
        plt.figure(figsize=(10, 8))
        plt.imshow(image,cmap=colormap,origin='lower',clim=(lo,up))
        plt.colorbar(orientation="horizontal")
        if title=='':
            #plt.title("$\lambda = $"+str(round(wavelengths,2))+" $\AA$",fontsize=18)
            plt.title(r"$\lambda = $" + str(round(wavelengths, 2)) + r" $\AA$", fontsize=18)

        else:
            plt.title(title,fontsize=18)
        plt.xlabel('xpix')
        plt.ylabel('ypix')
        plt.tight_layout()
        plt.show()
        
    elif np.ndim(image)==3:
        N = len(image)
        rows = (N+1) // 2
        
        fig, axes = plt.subplots(rows, 2, figsize=(12,10))
        axes = axes.flatten()
        
        for i in range(N):
            lo,up = np.nanpercentile(image[i],2),np.nanpercentile(image[i],98)
            im=axes[i].imshow(image[i], cmap=colormap,origin='lower',clim=(lo,up))
            axes[i].axis('off')
            axes[i].set_title(r"$\lambda = $"+str(round(wavelengths[i],2))+r" $\AA$",fontsize=18)
            
            axes[i].set_xlabel('xpix')
            axes[i].set_ylabel('ypix')
            fig.colorbar(im,orientation="horizontal")

        for j in range(N, len(axes)):
            axes[j].axis('off')

        
        
        plt.tight_layout()
        plt.show()


def plot_images(images, wavelengths, colormap='viridis', titles=None):
    
    N = len(images)  # Number of images
    rows = int(np.ceil(np.sqrt(N)))  # Number of rows
    cols = int(np.ceil(N / rows))  # Number of columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = np.array(axes).reshape(-1)  # Flatten to easily iterate
    
    for i in range(N):
        lo, up = np.nanpercentile(images[i], 2), np.nanpercentile(images[i], 98)
        im = axes[i].imshow(images[i], cmap=colormap, origin='lower', clim=(lo, up))
        axes[i].axis('off')

        # Set title (either custom or wavelength-based)
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=14)
        else:
            axes[i].set_title(r"$\lambda = {round(wavelengths[i],2)} \AA$", fontsize=14)

        # Add colorbar
        fig.colorbar(im, ax=axes[i], orientation="horizontal", fraction=0.046, pad=0.04)

    # Hide unused subplots if grid has extra slots
    for j in range(N, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


## visualization ##

def findWavelengths(arr, targets):
    
    def findWavelength(target):
        closest_index = min(range(len(arr)), key=lambda i: abs(arr[i] - target))
        return arr[closest_index], int(closest_index)

    if np.ndim(targets)==0:
        return findWavelength(targets)
    else:
        return [findWavelength(target) for target in targets]


## stacking ##

def stack_all(data):
    
    image_stack = np.array(data)
    
    return np.sum(image_stack, axis=0)

def stack(data,wave,number_images,central_wavelength):
    
    closest_indices = np.argsort(np.abs(wave - central_wavelength))[:number_images]
    
    selected_images = data[closest_indices]
    
    return stack_all(selected_images)


## image with median fluxes within certain wavelength bounds ##

def median(data,wave,bounds):
    
    min=bounds[0]
    max=bounds[1]
    
    x_len=len(data[0])
    y_len=len(data[0][0])
    
    res=np.zeros((x_len,y_len))
    
    N=len(data)
    
    
    #find bounds as indices
    i_min=int(findWavelengths(wave, min)[1])
    i_max=int(findWavelengths(wave, max)[1])
    
    print(i_min,i_max)
    print(wave[i_min:i_max+1])
    
    #print(data[i_min:i_max+1,:,:])
    
    for xx in range(0,x_len):
        for yy in range(0,y_len):    
            
            res[xx,yy]=np.median(data[i_min:i_max+1,xx,yy])
            
    return res

## image with average fluxes within certain wavelength bounds ##

def average(data,wave,bounds):
    
    min=bounds[0]
    max=bounds[1]
    
    x_len=len(data[0])
    y_len=len(data[0][0])
    
    res=np.zeros((x_len,y_len))
    
    N=len(data)
    
    
    #find bounds as indices
    i_min=int(findWavelengths(wave, min)[1])
    i_max=int(findWavelengths(wave, max)[1])
    
    print(i_min,i_max)
    print(wave[i_min:i_max+1])
    
    #print(data[i_min:i_max+1,:,:])
    
    for xx in range(0,x_len):
        for yy in range(0,y_len):    
            
            res[xx,yy]=np.average(data[i_min:i_max+1,xx,yy])
            
    return res


## median absolute deviation ##

def mad(data):
    median = np.nanmedian(data)
    
    return np.nanmedian(np.abs(data - median))


## staking spectra ##

def avg_spectra_of_region(data,z=0):
    wavelengths=len(data)
    region_area=len(data[0])*len(data[0][0])
    """spec_avg=np.zeros(wavelengths)
    for i in range(0,wavelengths):
        spec_avg[i]=np.sum(data[i,:,:])/(region_area)"""
    # sum along the first two axes (rows and columns of the region) and then average
    spec_avg = np.sum(data, axis=(1, 2)) / region_area
    
    return spec_avg

def median_spectra_of_region(data, z=0):
    spec_median = np.median(data, axis=(1, 2))
    return spec_median


## image signal to noise ##

def signaltonoise(image,noise): #this uses the error from MUSE as noise estimation
    image=np.array(image)
    noise=np.array(noise)
    result = np.where(
    (noise != 0) & (~np.isnan(image)) & (~np.isnan(noise)),  # Avoid division by zero & NaNs
    image / noise,
    np.nan  # Set problematic divisions to NaN
    )
    
    return result

## spectra signal to noise ##

def signaltonoise_spec(spec, uncertainty):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.asarray(spec) / np.asarray(uncertainty)
    
    return snr
## circurlar aperture ##

def circular_aperture(cube, x_center, y_center, radius):
    
    r = int(np.ceil(radius))
    pixels = []
    stacked_spectrum = np.zeros(cube.shape[0])  # initialize 1D spectrum array

    area=0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx**2 + dy**2 <= radius**2:
                area+=1
                x = int(x_center + dx)
                y = int(y_center + dy)

                # Check bounds
                if 0 <= x < cube.shape[2] and 0 <= y < cube.shape[1]:
                    pixels.append((x, y))
                    stacked_spectrum += cube[:, y, x]  # note: y is row, x is column
    
    stacked_spectrum=stacked_spectrum/area
    
    return stacked_spectrum#, pixels


## binning ##

def sum_submatrix(matrix,row_i,col_i,row_width,col_width): #row_i,col_i,number of rows, number of cols
    row_width-=1
    #print(matrix[row_i:row_i+row_width+1, col_i:col_i+col_width])
    return np.sum(matrix[row_i:row_i+row_width+1, col_i:col_i+col_width])

def binning(image, pix_width): #this ignores the existence of NaNs
    
    
    x_len=len(image[0])
    y_len=len(image)
    
    
    matrix=[]
    for j in range(0,y_len):
        
        if j%pix_width==0: #and j!=0:
            row=[]
            for i in range(0,x_len):
                if i%pix_width==0: #and i!=0:
                    
                    sum=sum_submatrix(image,j,i,pix_width,pix_width)
                    
                    row.append(sum/(pix_width*pix_width))
                    
            
            matrix.append(row)
                    
                
    return matrix


## voronoi binning ##


def voronoi(flux_map,noise_values,target_snr=20,plots=False):



    ny, nx = flux_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny))
    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()
    flux_values = flux_map.ravel()

    target_snr

    out = voronoi_2d_binning(x_coords, y_coords, flux_values, noise_values, target_snr, plot=plots, quiet=True)

    bin_num = out[0]
    x_bin = out[1]
    y_bin = out[2]
    sn_bin = out[3]
    n_pixels = out[4]

    if plots==True:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, c=bin_num, cmap='Blues_r', s=5)
        plt.colorbar(label="Bin ##index")
        plt.title(f"Voronoi Binning using target SNR={target_snr}")
        plt.show()

    binned_data = np.zeros(len(flux_values))

    for i in range(0,len(n_pixels)):
        bin_mask = bin_num == i
        binned_data[bin_mask] = np.nanmedian(flux_values[bin_mask])
        
    binned_data = np.transpose(binned_data.reshape(ny,nx))

    return binned_data



## fitting ##

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def gaussian_error(params, cov_matrix): #returns flux error (sigma_f)

    A, mu, sigma = params

    #df_dA = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    df_dA = lambda x:  np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    #df_dmu = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) * (x - mu) / sigma**2
    df_dmu = lambda x: A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) * (x - mu) / sigma**2
    
    #df_dsigma = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) * (x - mu)**2 / sigma**3
    df_dsigma = lambda x: A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) * (x - mu)**2 / sigma**3
    
    sigma_A = np.sqrt(cov_matrix[0,0])
    sigma_mu = np.sqrt(cov_matrix[1,1])
    sigma_sigma = np.sqrt(cov_matrix[2,2])

    err = lambda x: np.sqrt((df_dA(x) * sigma_A)**2 + (df_dmu(x) * sigma_mu)**2 + (df_dsigma(x) * sigma_sigma)**2)

    return err #np.sqrt((df_dA * sigma_A)**2 + (df_dmu * sigma_mu)**2 + (df_dsigma * sigma_sigma)**2)


def polynomial(x, c1, c2, c3, c4, c5):
    return  c1 + c2 * x + c3 * x**2 + c4 * x**3 + c5 * x**4

def polynomial_line(x, c1, c2):
    return  c1 + c2 * x

def polynomial_error(x, params, cov_matrix): #returns flux error (sigma_c)

    c1, c2, c3, c4, c5 = params

    df_dc1 = 1
    df_dc2 = x
    df_dc3 = x**2
    df_dc4 = x**3
    df_dc5 = x**4
    
    sigma_c1 = np.sqrt(cov_matrix[0,0])
    sigma_c2 = np.sqrt(cov_matrix[1,1])
    sigma_c3 = np.sqrt(cov_matrix[2,2])
    sigma_c4 = np.sqrt(cov_matrix[3,3])
    sigma_c5 = np.sqrt(cov_matrix[4,4])
    
    return np.sqrt((df_dc1 * sigma_c1)**2 + (df_dc2 * sigma_c2)**2 + (df_dc3 * sigma_c3)**2 + (df_dc4 * sigma_c4)**2 + (df_dc5 * sigma_c5)**2)

"""def gaussian_polynomial(x, A, mu, sigma, c1, c2, c3):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c1 + c2 * x + c3 * x**2"""

def background(x,c0,c1):
    return c0 + c1*x

def three_gaussian_poly(x, c0, c1, A1, sigma1, A2, mu2, sigma2, A3, sigma3,shift1=14.8,shift2=20.6):
    bg =  background(x,c0,c1)#+ c2*x**2
    mu1=mu2-shift1
    mu3=mu2+3
    peak1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    peak2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    peak3 = A3 * np.exp(-((x - mu3)**2) / (2 * sigma3**2))
    return bg + peak1 + peak2 + peak3

def two_gaussian_poly(x, c0, c1, A1, mu1, sigma1, A2, mu2,sigma2,shift):
    bg =  background(x,c0,c1)
    
    mu2=mu1+shift
    peak1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    peak2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))

    return bg + peak1 + peak2

def two_gaussian_poly_nobg(x, A1, mu1, sigma1, A2, mu2,sigma2,shift=4.5):
 
    mu2=mu1+shift
    peak1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    peak2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))

    return peak1 + peak2


## chopping data to window view ##

def chop_data(x,y,x_min,x_max,err=[]):
    
    mask = (x >= x_min) & (x <= x_max)
    
    
    if len(err)<1: #if no errors are inputted
        x_chopped, y_chopped = x[mask], y[mask]
        return x_chopped, y_chopped

    else:
        x_chopped, y_chopped, err_chopped = x[mask], y[mask], err[mask]
        return x_chopped, y_chopped, err_chopped


## remove duplicate points from theoretical spectra

def average_duplicate_wavelengths(wavelength, flux):
    
    df = pd.DataFrame({'wavelength': wavelength, 'flux': flux})
    df_unique = df.groupby('wavelength', as_index=False).mean()
    return df_unique['wavelength'].values, df_unique['flux'].values


## cosine kernel ##

def cosine_kernel(size):
    x = np.linspace(-np.pi, np.pi, size)
    kernel = (1 + np.cos(x))/2  
    return kernel / np.sum(kernel)

## smoothing the spectra ##

def smooth_spectra(y,kernel_size):

    kernel = cosine_kernel(kernel_size)
    y_smooth = convolve1d(y, kernel, mode='nearest')
    return y_smooth

def continuum(x,y,mode="both"):

    ## selecting points for continuum
    x_continuum,y_continuum=filterout_peaks(x,y,mode)
    
    k = np.mean(x_continuum)
    x_continuum -= k

    if len(y_continuum)==0:
        plt.plot(x,y)
        plt.show()
        print("Couldn't find continuum")
        return
        
    ## fitting the selected points
    y_err=np.full(len(y_continuum), mad(y_continuum))#np.sqrt(np.median(ebulge, axis=(1, 2)));
    
    initial_guess = [100,1,1,1,1]
    params, covariance = curve_fit(polynomial, x_continuum, y_continuum, p0=initial_guess, sigma=y_err, absolute_sigma=True)
    fit = lambda x: polynomial(x-k,*params)

    err_cont = lambda x: polynomial_error(x-k, params, covariance)
    return fit, x_continuum+k, y_continuum, err_cont

def continuum_line(x,y,mode="both"):

    ## selecting points for continuum
    x_continuum,y_continuum=filterout_peaks(x,y,mode)
    
    k = np.mean(x_continuum)
    x_continuum -= k

    if len(y_continuum)==0:
        plt.plot(x,y)
        plt.show()
        print("Couldn't find continuum")
        return
        
    ## fitting the selected points
    y_err=np.full(len(y_continuum), mad(y_continuum))
    
    initial_guess = [100,1]
    params, covariance = curve_fit(polynomial_line, x_continuum, y_continuum, p0=initial_guess, sigma=y_err, absolute_sigma=True)
    fit = lambda x: polynomial_line(x-k,*params)

    err_cont = lambda x: polynomial_error(x-k, params, covariance)
    return fit, x_continuum+k, y_continuum, err_cont

## filtering out peaks ##

"""def filterout_peaks(x,y,low=30,high=70):
    Q1 = np.percentile(y, low)
    Q3 = np.percentile(y, high)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    mask= y<threshold
    filtered_x=(x)[mask]
    filtered_y=(y)[mask]
    #filtered_y = y[y < threshold]  # Remove peaks
    
    return filtered_x, filtered_y"""


def filterout_peaks(x, y, mode="peaks", low=30, high=70):
    
    Q1 = np.percentile(y, low)
    Q3 = np.percentile(y, high)
    IQR = Q3 - Q1
    upper_threshold = Q3 + 1.5 * IQR  # Threshold for peaks
    lower_threshold = Q1 - 1.5 * IQR  # Threshold for dips

    if mode == "peaks":
        mask = y < upper_threshold  # Remove peaks
    elif mode == "dips":
        mask = y > lower_threshold  # Remove dips
    else:  # mode == "both"
        mask = (y > lower_threshold) & (y < upper_threshold)  # Remove both peaks and dips

    filtered_x = x[mask]
    filtered_y = y[mask]

    return filtered_x, filtered_y

## EW ##

def EW_parametric(x,y,MUSE_err,cont,cont_error,method,plots,fit="Halpha",central_wavelength=6623.2630764): #
    
    if fit=="Halpha":
        max=np.argmax(y)
        bound1,bound2=x[max]-10,x[max]+10
    else:
        bound1,bound2=central_wavelength-50,central_wavelength+50

    err=0

    if plots==True:
        plt.figure(figsize=(8, 5))
    
    if method==0: # computing the EW from a fit to (flux-continuum)
        x,y=chop_data(x,y,bound1,bound2)
        # removing the continuum from the flux data
        flux_reduced = cont(x)-y
        
        y_err=np.full(len(flux_reduced), mad(flux_reduced))#np.sqrt(np.median(ebulge, axis=(1, 2)));

        if plots==True:
            plt.fill_between(x,flux_reduced - y_err, flux_reduced + y_err, color='blue', alpha=0.1, label="Uncertainty")
        
        x_fit = np.linspace(np.min(x), np.max(x), 100)

        if fit=="Halpha":
            initial_guess = [np.max(flux_reduced), np.mean(x), np.std(x)]
            params, covariance = curve_fit(gaussian, x, flux_reduced, p0=initial_guess, sigma=y_err, absolute_sigma=True)
            A_fit, mu_fit, sigma_fit = params
            
            y_fit = gaussian(x_fit, params[0], params[1], params[2])
            flux_reduced_gaussian = lambda x: gaussian(x,*params)
            gaussian_err=gaussian_error(params, covariance)
        elif fit=="Na":
            height=np.max(flux_reduced)
            initial_guess = [height,central_wavelength,1,height*0.7,2,0.5,4.5]
            params, covariance = curve_fit(two_gaussian_poly_nobg, x, flux_reduced, p0=initial_guess,maxfev = 8000)#input errors!
            
            y_fit = two_gaussian_poly_nobg(x_fit, *params)
            flux_reduced_gaussian = lambda x: two_gaussian_poly_nobg(x,*params)
            gaussian_err=0#??????
            
        

        
        

        xx=np.linspace(bound1,bound2, 100)  # Generate 100 new points
        
        delta=xx[2]-xx[1]
        
        val=0
        err=0
        MUSE_err=y_err[1]
        for xi in xx:
            val+=delta*(flux_reduced_gaussian(xi))/cont(xi)
            #err+=delta*(np.sqrt( (-1/cont(xi))**2 * MUSE_err**2 + ((cont(xi)-flux_reduced_gaussian(xi))/cont(xi)**2)**2 * cont_error(xi)**2 + (1/cont(xi))**2 * gaussian_err(xi)**2 ))
            

        
    elif method==1: # computing the EW from a fit to flux (continuum+gaussian)

        
        y_err=np.full(len(y), mad(y))#np.sqrt(np.median(ebulge, axis=(1, 2)));
        if plots==True:
            plt.fill_between(x,y - y_err, y + y_err, color='blue', alpha=0.4, label="Uncertainty",zorder=4)

        if fit=="Halpha":
            #c0, c1, A1, sigma1, A2, mu2, sigma2, A3, sigma3
            initial_guess = [1, 1, np.max(y)/3, 3, np.max(y), central_wavelength, 5, np.max(y)/2, 3]
            params, covariance = curve_fit(three_gaussian_poly, x, y, p0=initial_guess, maxfev=8000, sigma=y_err, absolute_sigma=True)
            c0_fit, c1_fit, A1_fit, sigma1_fit, A2_fit, mu2_fit, sigma2_fit, A3_fit, sigma3_fit = params
                    
            x_fit = np.linspace(np.min(x), np.max(x), 500)
            y_fit = three_gaussian_poly(x_fit, *params)

            
            # flux_reduced_gaussian is (-1)*gaussian of the main peak
            flux_reduced_gaussian = lambda xi: -1*gaussian(xi, A2_fit, mu2_fit,sigma2_fit)
            
            flux_reduced = flux_reduced_gaussian(x)

            cont = lambda xi: background(xi,c0_fit,c1_fit)#three_gaussian_poly(x,*params) + flux_reduced_gaussian(x)
        elif fit=="Na":
            height=np.max(y)-np.min(y)
            initial_guess = [2,0.5,height,central_wavelength,1,height,2,1,4.5]
            params, covariance = curve_fit(two_gaussian_poly, x, y, p0=initial_guess)#input errors!
            
            fit_exp = lambda xi: two_gaussian_poly(xi,*params)
            #print("!!!! lenx",len(x))
            x_fit = np.linspace(np.min(x), np.max(x), 500)
            y_fit = two_gaussian_poly(x_fit, *params)

            c0_fit, c1_fit = params[0],params[1]
            cont = lambda xi: background(xi,c0_fit,c1_fit)
            
            flux_reduced_gaussian = lambda xi: cont(xi) - fit_exp(xi)

            flux_reduced = flux_reduced_gaussian(x)
            


        
        xx=np.linspace(bound1,bound2, 100)  # Generate 100 new points
        
        delta=xx[2]-xx[1]
        
        val=0
    
        for xi in xx:
            
            val+=delta*(flux_reduced_gaussian(xi))/cont(xi)
            #err+=delta*(np.sqrt( (1/cont(xi))**2 * err_flux_reduced **2 + ((flux_reduced_gaussian(xi))/cont(xi)**2)**2 * cont_error(xi)**2 + )

    
    if plots==True:
    
        #plot of error propagated to f
        #plt.fill_between(x, cont(x)-y-gaussian_error(x, params, covariance),cont(x)-y+gaussian_error(x, params, covariance), alpha=0.3, color='red', label="propagated error of f")
        
        
        plt.scatter(x, y, label="Original data", color="black",s=10)
        plt.plot(x,cont(x), label="Continuuum")
        
        plt.scatter(x, flux_reduced, label="Continuum-Flux", color="red",s=3)
        plt.plot(x,flux_reduced_gaussian(x),color="yellow",label="cont-fit",linewidth=0.5)
        plt.plot(x_fit, y_fit, label="fit")
        if fit=="Halpha":
            plt.fill_between(x_fit, y_fit, alpha=0.3, color='gray', label="Integral")
        if fit=="Na":
            plt.fill_between(x, flux_reduced_gaussian(x), alpha=0.3, color='gray', label="Integral")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.legend()
        plt.title("Finding EW")
        plt.show()


    return val,err
    
    


def error_non_parametric(Delta,cont,sigma_c,g,sigma_f):

    #g==(c-y)/c
    #computing sigma_g
    
    sigma_g=[]

    for i in range(0,len(cont)):
        sigma_g.append(np.sqrt((sigma_f/cont[i])**2 + (g[i]* sigma_c / cont[i])**2))
        
    sum=0

    for i in range(0,len(sigma_g)-1):
        sum += (Delta/2)**2 * (sigma_g[i]**2 + sigma_g[i+1]**2)

    noise= mad(cont)
    err_integral = np.sqrt(sum)

    return err_integral

## Maps ##
#parametric#
def EW_map_parametric(cube_region,wave,MUSE_err,central_wavelength,mode="peaks",kernel_size=3,method=0, plots=False,fit="Halpha"):
        
    
    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    ew_map=np.zeros((x_len, y_len))
    error_map=np.zeros((x_len, y_len))
    
    x_min=central_wavelength-50
    x_max=central_wavelength+50
            
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]
            
            # chop data
            x_chopped,y_chopped=chop_data(wave,spec,x_min,x_max)
            
            
            # smooth data
            y_smooth=smooth_spectra(y_chopped,kernel_size)
            
            # fit to continuum
            if fit=="Halpha":
                aux=continuum(x_chopped,y_smooth,mode)
            elif fit=="Na":
                aux=continuum_line(x_chopped,y_smooth,mode)

            if(aux==None): # skipping the i,j pixel in case we cannot find the continuum
                map[i,j]=np.nan
                continue 
            else:
                continuum_fit=aux[0]
                x_cont=aux[1]
                y_cont=aux[2]
                continuum_error=aux[3]
            
            y_continuum_fit = continuum_fit(x_chopped)          
            
            
            # measuring EW
            """if i==21 and j==15:
                plots=True"""
            ew,err=EW_parametric(x_chopped,y_smooth,MUSE_err,continuum_fit,continuum_error,method,plots,fit,central_wavelength=central_wavelength)
            print(np.min(x_chopped),np.max(x_chopped))
            print("EW = %.3f"%ew," +/- %.3f"%err," at (i,j)=",i,",",j)
            
            ew_map[i,j]=ew
            error_map[i,j]=err
            
    return ew_map, error_map

def EW_map_non_parametric(cube_region,wave,central_wavelength,mode,kernel_size=3,plots=False,velocity_window=1000):

    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    ew_map=np.zeros((x_len, y_len))
    err_map=np.zeros((x_len, y_len))
    
    
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]
            
            x_chopped,y_chopped=chop_data(wave,spec,central_wavelength-50,central_wavelength+50)
            
            # smooth data
            y_smooth=smooth_spectra(y_chopped,6) # this is not doing anything


            # compute continuum with kernel
            x_cont,y_cont=filterout_peaks(x_chopped,y_smooth,mode)
            
            kernel = cosine_kernel(kernel_size)
            cont = convolve1d(y_cont, kernel, mode='nearest')
            interp=interp1d(x_cont, cont, kind='cubic')
            
            #max=np.argmax(y_smooth)
            #b1=x_chopped[max]-10
            #b2=x_chopped[max]+10
            #b1=central_wavelength-10
            #b2=central_wavelength+10
            b1=central_wavelength*(1-velocity_window/(3*10**5))
            b2=central_wavelength*(1+velocity_window/(3*10**5))

            x,y=chop_data(x_chopped,y_smooth,b1,b2)
            
            continuum = interp(x)

            
            # Compute the excess intensity above the continuum            
            excess_intensity = (continuum-y)/continuum

            # Integrate the excess intensity (area over the continuum)
            #area_over_continuum = trapezoid(excess_intensity, x)
            
            # Interpolating before integrating, so that we have more points
            
            g = interp1d(x, excess_intensity, kind='cubic')

            xx=np.linspace(np.min(x),np.max(x), 100)  # Generate 100 new points
            excess_intensity = g(xx)
            

            # Integrate the excess intensity (area over the continuum)
            area_over_continuum = trapezoid(excess_intensity, xx)


            # EW error computation
            err_cont=mad(interp(xx))
            err_f=mad(y)
            err=error_non_parametric(xx[2]-xx[1],interp(xx),err_cont,g(xx),err_f)

            """if i==21 and j==15:
                plots=True"""


            if plots==True:
                plt.scatter(x_cont,y_cont,color="yellow",label="points for continuum")
                plt.plot(x_chopped,y_smooth,label="smooth spectra")
                plt.plot(x, continuum-y, label="cont-spec", color="black")
                plt.plot(x, continuum, label="Continuum", linestyle="dashed", color="red")
                plt.fill_between(x, (continuum-y), 0, alpha=0.3, color="green", label="Excess area")
                plt.xlabel("Wavelength")
                plt.ylabel("Intensity")
                plt.legend()
                plt.show()

            
            ew=area_over_continuum
            print(f"Integral of area over continuum divided by continuum: {area_over_continuum:.3f} +/- {err:.3f} at (i,j)=",i,", ",j)
            
            ew_map[i,j]=ew
            err_map[i,j]=err
    return ew_map,err_map



def velocity(x,y,cont,lambda_rest):

   
    c = 3*10**5 # in km/s

    x,y=chop_data(x,y,lambda_rest-4,lambda_rest+4)


    num,denom=0,0
    for i in range(len(x)):
        denom += 1-y[i]/cont(x[i])
        num +=  (1-y[i]/cont(x[i])) * x[i]

    v = c/lambda_rest * (num/denom - lambda_rest)

    """if v>0:
        plt.plot(x,y)
        plt.axvline(x=num/denom)
        plt.show()"""

    return v

# computing the EW from a fit to flux (continuum+gaussian)

def velocity_parametric(x,y,MUSE_err,rest_wavelenght,i,j):


    max=np.argmax(y)
    bound1,bound2=x[max]-10,x[max]+10
    err=0    


    y_err=np.full(len(y), mad(y))#np.sqrt(np.median(ebulge, axis=(1, 2)));


    try:
        #c0, c1, A1, sigma1, A2, mu2, sigma2, A3, sigma3
        initial_guess = [1, 1, np.max(y)/3, 5, np.max(y), rest_wavelenght, 3, np.max(y)/3, 5]
        params, covariance = curve_fit(three_gaussian_poly, x, y, p0=initial_guess, maxfev=10000, sigma=y_err, absolute_sigma=True)
        c0_fit, c1_fit, A1_fit, sigma1_fit, A2_fit, mu2_fit, sigma2_fit, A3_fit, sigma3_fit = params
                
        x_fit = np.linspace(np.min(x), np.max(x), 500)
        y_fit = three_gaussian_poly(x_fit, *params)


        val = 3*10**5 * (mu2_fit-rest_wavelenght) / rest_wavelenght
        #c/lambda_rest * (num/denom - lambda_rest)
        err = 3*10**5 / rest_wavelenght * (np.sqrt(covariance[6,6]))
        
        return val,err

    except RuntimeError:
        print(f"Iteration {i}: Fit failed, skipping...")
        return np.nan,np.nan

    
    
# non parametric map
def velocity_map(cube_region,wave,lambda_rest,kernel_size=3,mode="both"):

    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    map=np.zeros((x_len, y_len))
    
    
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]

            # chop data
            b1=lambda_rest-40
            b2=lambda_rest+40
            x_chopped,y_chopped=chop_data(wave,spec,b1,b2)
                        
            # smooth data
            y_smooth=smooth_spectra(y_chopped,3)
                        
            # fit to continuum
            #cont=continuum(x_chopped,y_smooth,mode="peaks")[0]

            # compute continuum with kernel
            x_cont,y_cont=filterout_peaks(x_chopped,y_smooth,mode)
            
            kernel = cosine_kernel(kernel_size)
            cont = convolve1d(y_cont, kernel, mode='nearest')
            interp=interp1d(x_cont, cont, kind='cubic')
            
            continuum = interp
                        
            
            map[i,j]=velocity(x_chopped,y_chopped,continuum,lambda_rest)

    return map

#parametric#
def velocity_map_parametric(cube_region,wave,MUSE_err,rest_wavelenght,kernel_size=3):
        
    
    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    v_map=np.zeros((x_len, y_len))
    error_map=np.zeros((x_len, y_len))
    
    x_min=rest_wavelenght-40
    x_max=rest_wavelenght+40
            
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]
            
            # chop data
            x_chopped,y_chopped=chop_data(wave,spec,x_min,x_max)
            
            # smooth data
            y_smooth=smooth_spectra(y_chopped,kernel_size)
                   
            # measuring velocity
            vel,err=velocity_parametric(x_chopped,y_smooth,MUSE_err,rest_wavelenght,i,j)

            print("velocity = %.3f"%vel," +/- %.3f"%err," at (i,j)=",i,",",j)
            
            v_map[i,j]=vel
            error_map[i,j]=err
            
    return v_map, error_map


# matching MUSE stars to Gaia

def match_gaia(sources,header,ra,dec,width=0.2,height=0.2):
    

    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"#edr3 or dr2
    Gaia.ROW_LIMIT = -1

    ## get ra,dec from WCS astrometry header
    wcs_header = WCS(header)
    coords = wcs_header.pixel_to_world(sources[:,0],sources[:,1], np.full(len(sources), 0))

    print(coords)
    ## get Gaia catalog around center ra/dec values
    cencoord = SkyCoord(ra=ra,dec=dec,unit=(u.deg,u.deg),frame='icrs')


    width,height = u.Quantity(width, u.deg),u.Quantity(height, u.deg)
    gaia_stars = Gaia.query_object_async(coordinate=cencoord, width=width, height=height)
    gaia_coords = SkyCoord(ra=gaia_stars['ra'],dec=gaia_stars['dec'])

    ## match catalogs
    gidx, gd2d, gd3d = coords[0].match_to_catalog_sky(gaia_coords)
    gbestidx=(gd2d.deg < 0.0008)                         #<0.00015deg=0.54''

    ## output variables
    star_ra,star_dec = np.zeros(len(sources),dtype=float),np.zeros(len(sources),dtype=float)
    star_ra[:],star_dec[:] = np.nan,np.nan
    star_ra[gbestidx] = gaia_stars['ra'][gidx[gbestidx]]
    star_dec[gbestidx] = gaia_stars['dec'][gidx[gbestidx]]
    star_par,star_parer = np.zeros(len(sources),dtype=float)*np.nan,np.zeros(len(sources),dtype=float)*np.nan
    star_par[gbestidx] = gaia_stars['parallax'][gidx[gbestidx]]
    star_parer[gbestidx] = gaia_stars['parallax_error'][gidx[gbestidx]]

    return star_ra,star_dec,star_par,star_parer

def gaia_parameters(matched_ras,matched_decs):
    star_ra=matched_ras
    star_dec=matched_decs
    
    parallax_array = np.full(len(star_ra), np.nan)
    parallax_err_array = np.full(len(star_ra), np.nan)
    eff_t_array = np.full(len(star_ra), np.nan)
    surface_g_array = np.full(len(star_ra), np.nan)
    metallicity_array = np.full(len(star_ra), np.nan)
    mean_mag = np.full(len(star_ra), np.nan)

    
    for i in range(len(star_ra)):
        
        ra = star_ra[i]
        dec = star_dec[i]

        if np.isnan(ra) or np.isnan(dec):
            continue  # skip if either value is NaN

        print(f"Searching for star with RA = {ra}, Dec = {dec}")

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        radius_deg = (1 * u.arcsec).to(u.deg).value

        query = f"""
        SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.parallax, g.parallax_error,
           a.teff_gspphot, a.logg_gspphot, a.mh_gspphot
        FROM gaiadr3.gaia_source AS g
        LEFT JOIN gaiadr3.astrophysical_parameters AS a
          ON g.source_id = a.source_id
        WHERE 1=CONTAINS(
          POINT('ICRS', g.ra, g.dec),
          CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_deg})
        )
        """



        try:
            job = Gaia.launch_job_async(query)
            result = job.get_results()
            print("warning")
            print(result.columns)

            if len(result) == 0:
                print("No match found.")
            else:
                parallax_array[i] = result[0]["parallax"]
                parallax_err_array[i] = result[0]["parallax_error"]
                eff_t_array[i] = result[0]["teff_gspphot"]
                surface_g_array[i] = result[0]["logg_gspphot"]
                metallicity_array[i] = result[0]["mh_gspphot"]
                mean_mag[i] = result[0]["phot_g_mean_mag"]
                
                print(f"Parallax: {result[0]['parallax']} ± {result[0]['parallax_error']}")
                print(f"Effective temperature: ", result[0]['teff_gspphot'])
                print(f"Surface gravity log(g): ", result[0]['logg_gspphot'])
                print(f"Metalicity: ", result[0]['mh_gspphot'])#feh_gspspec
                print(f"Mean magnitude: ", result[0]['phot_g_mean_mag'])

        except Exception as e:
            print(f"Query failed for index {i}: {e}")

        print(" ")
    return parallax_array,parallax_err_array,eff_t_array,surface_g_array,metallicity_array, mean_mag

def EW_point_sources(cube, sources, wave, na_rest,radius=0,plots=False):
    EW_array=[]
    EW_err_array=[]
    for i in range(0,len(sources)):
        
        y_pos=sources[i][1]
        x_pos=sources[i][0]

        #data=cube[:,y_pos,x_pos]
        data=circular_aperture(cube,x_pos, y_pos, radius)#new


        
        x_chopped,y_chopped=chop_data(wave,data,na_rest-80,na_rest+80)

        y_smooth=smooth_spectra(y_chopped,kernel_size=3)
        # continuum
        x,y=x_chopped,y_smooth
        x_cont,y_cont=filterout_peaks(x,y,mode="both")

        noise = mad(y_cont)

        kernel_size=60
        kernel = cosine_kernel(kernel_size)
        cont = convolve1d(y_cont, kernel, mode='nearest')
        interp=interp1d(x_cont, cont, kind='cubic')



        v=600
        bound1=na_rest*(1-v/(3*10**5))

        bound2=na_rest*(1+v/(3*10**5))
        x,y=chop_data(x,y,bound1,bound2)

        cont = interp(x)

        # Compute the excess intensity above the continuum
        excess_intensity = (cont-y)/cont
        err_f=mad(y)
        g = interp1d(x, excess_intensity, kind='cubic')

        if plots==True:
            plt.plot(x,excess_intensity,label="integral")
            #plt.plot(x_chopped,y_chopped,label="spectra")
            plt.show()
        # Integrate the excess intensity (area over the continuum)
        area_over_continuum = trapezoid(excess_intensity, x)
        #continuum_summed = simps(continuum_fit(x), x)

        # Compute uncertainty

        err_cont=mad(interp(x))
        err=error_non_parametric(x[2]-x[1],interp(x),err_cont,g(x),err_f)

        #err += (x[2]-x[1]) * noise  * np.sqrt(len(x)) # adding noise estimate to the error estimate
        #err=np.sqrt(noise)

        print(f"EW= {area_over_continuum:.2f}"," +/- ", err)
        EW_array.append(area_over_continuum)
        EW_err_array.append(err)
    return EW_array, EW_err_array



def EW_theoretical_spectra(wavelength,flux, na_rest,plots=False):
    EW_array=[]
    EW_err_array=[]


    x_chopped,y_chopped=chop_data(wavelength,flux,na_rest-80,na_rest+80)

    
    x_chopped,y_chopped=average_duplicate_wavelengths(x_chopped,y_chopped)
    # continuum
    x,y=x_chopped,y_chopped
    x_cont,y_cont=filterout_peaks(x,y,mode="both")

    kernel_size=60
    kernel = cosine_kernel(kernel_size)
    cont = convolve1d(y_cont, kernel, mode='nearest')

    interp=interp1d(x_cont, cont, kind='cubic')



    v=500
    bound1=na_rest*(1-v/(3*10**5))

    bound2=na_rest*(1+v/(3*10**5))
    x,y=chop_data(x,y,bound1,bound2)

    cont = interp(x)

    # Compute the excess intensity above the continuum
    excess_intensity = (cont-y)/cont
    err_f=mad(y)
    g = interp1d(x, excess_intensity, kind='cubic')

    if plots==True:
        plt.plot(x,excess_intensity)
        plt.show()
    # Integrate the excess intensity (area over the continuum)
    area_over_continuum = trapezoid(excess_intensity, x)
    #continuum_summed = simps(continuum_fit(x), x)

    # Compute uncertainty
    err_cont=mad(interp(x))
    err=error_non_parametric(x[2]-x[1],interp(x),err_cont,g(x),err_f)

    print(f"EW= {area_over_continuum:.2f}"," +/- ", err)
    EW_array.append(area_over_continuum)
    EW_err_array.append(err)
    return EW_array, EW_err_array



def generate_spectra(model,stars_data,figures=False):

    spectra=[]
    RES=20000
    for i in range(0,len(stars_data)):
        model_param = {'teff':stars_data.iloc[i]['teff'], 'logg':stars_data.iloc[i]['logg'], 'feh':stars_data.iloc[i]['met'], 'distance':1/stars_data.iloc[i]['parallax']}

        model_box = model.get_model(model_param=model_param, spec_res=RES)
        model_ext = model.get_model(model_param=model_param, spec_res=RES)

        if figures==True:
            print(model_box.open_box())

            fig = plot_spectrum(boxes=[model_box, model_ext],
                            filters=['MKO/NSFCam.J', 'MKO/NSFCam.H', 'MKO/NSFCam.K', 'MKO/NSFCam.Lp', 'MKO/NSFCam.Mp'],
                            legend={'loc': 'upper right', 'frameon': False, 'fontsize': 8.5},
                            figsize=(7., 3.),
                            output=None)

        b1,b2=(5890-30)*0.0001,(5896+30)*0.0001

        x=model_box.wavelength
        #x*=10000
        y=model_box.flux

        spectra.append([x,y])

    return spectra


def compute_ebv_gas(Ha_flux, Hb_flux, law='calzetti'):
    """
    Calculate E(B-V) for the ionized gas using the Balmer decrement.

    Parameters:
    - Ha_flux: Observed H-alpha flux
    - Hb_flux: Observed H-beta flux
    - law: 'calzetti' or 'cardelli'

    Returns:
    - E(B-V) value (float)
    """
    # Intrinsic ratio from case B recombination
    intrinsic_ratio = 2.86

    # Extinction coefficients
    if law == 'calzetti':
        k_Ha = 3.33
        k_Hb = 4.60
    elif law == 'cardelli':
        k_Ha = 2.535
        k_Hb = 3.609
    else:
        raise ValueError("Choose 'calzetti' or 'cardelli' for extinction law")

    # Observed ratio
    observed_ratio = Ha_flux / Hb_flux

    # Avoid log of negative or zero
    if observed_ratio <= intrinsic_ratio:
        return 0.0

    # Compute E(B-V)
    ebv = (2.5 / (k_Hb - k_Ha)) * np.log10(observed_ratio / intrinsic_ratio)
    return ebv
