import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.integrate import simps
from scipy.interpolate import interp1d


import matplotlib.animation as animation


## visualization ##

def plot_image(image, wavelengths, index, colormap,title=''):
    if np.ndim(image)==2:
        lo,up = np.nanpercentile(image,2),np.nanpercentile(image,98)
        plt.figure(figsize=(10, 8))
        plt.imshow(image,cmap=colormap,origin='lower',clim=(lo,up))
        plt.colorbar(orientation="horizontal")
        if title=='':
            plt.title("$\lambda = $"+str(round(wavelengths,2))+" $\AA$",fontsize=18)
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
            axes[i].set_title("$\lambda = $"+str(round(wavelengths[i],2))+" $\AA$",fontsize=18)
            
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
            axes[i].set_title(f"$\lambda = {round(wavelengths[i],2)} \AA$", fontsize=14)

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

## fitting ##

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


## chopping data to window view ##

def chop_data(x,y,x_min,x_max):
    
    
    mask = (x >= x_min) & (x <= x_max)
    x_chopped, y_chopped = x[mask], y[mask]
    
    return x_chopped, y_chopped


## cosine kernel ##

def cosine_kernel(size):
    x = np.linspace(-np.pi, np.pi, size)
    kernel = (1 + np.cos(x))/2  
    return kernel / np.sum(kernel)

## smoothing the spectra ##

def smooth_spectra(y,kernel_size):

    kernel = cosine_kernel(kernel_size)
    y_smooth = convolve(y, kernel, mode='same')
    return y_smooth

def continuum(x,y,threshold=100):#threshold = 100  was based on experimentation...

    ## selecting points for continuum
    #continuum_mask = (y_chopped <= y_smooth * (1+threshold))  # points of smooothed spectra
    continuum_mask = y<threshold  # points of smooothed spectra
    x_continuum = (x)[continuum_mask]
    y_continuum = (y)[continuum_mask]


    if len(y_continuum)==0:
        plt.plot(x,y)
        plt.show()
        print("Couldn't find continuum")
        return
        
    ## fitting the selected points
    p_coeffs = np.polyfit(x_continuum, y_continuum, 4)
    fit = np.poly1d(p_coeffs)
    
    return fit, x_continuum, y_continuum
