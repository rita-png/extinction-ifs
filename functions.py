import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.integrate import trapz
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


def gaussian_error(x, params, cov_matrix): #returns flux error (sigma_f)

    A, mu, sigma = params

    df_dA = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    df_dmu = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) * (x - mu) / sigma**2
    df_dsigma = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) * (x - mu)**2 / sigma**3
    
    sigma_A = cov_matrix[0,0]
    sigma_mu = cov_matrix[1,1]
    sigma_sigma = cov_matrix[2,2]

    return np.sqrt(df_dA * sigma_A + df_dmu * sigma_mu + df_dsigma * sigma_sigma)


def polynomial(x, c1, c2, c3, c4, c5):
    return  c1 + c2 * x + c3 * x**2 + c4 * x**3 + c5 * x**4

"""def gaussian_polynomial(x, A, mu, sigma, c1, c2, c3):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c1 + c2 * x + c3 * x**2"""

def three_gaussian_poly(x, c0, c1, c2, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    background = c0 + c1*x + c2*x**2
    peak1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    peak2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    peak3 = A3 * np.exp(-((x - mu3)**2) / (2 * sigma3**2))
    return background + peak1 + peak2 + peak3

def EW_error():


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

def continuum(x,y):

    ## selecting points for continuum
    x_continuum,y_continuum=filterout_peaks(x,y)

    if len(y_continuum)==0:
        plt.plot(x,y)
        plt.show()
        print("Couldn't find continuum")
        return
        
    ## fitting the selected points
    y_err=np.full(len(y_continuum), mad(y_continuum))#np.sqrt(np.median(ebulge, axis=(1, 2)));
    """weights = 1 / (y_err + 10**(-6))
    #print("weigths, ", weights)
    p_coeffs = np.polyfit(x_continuum, y_continuum, 4), w=weights, cov=True)
    fit = np.poly1d(p_coeffs)"""
    initial_guess = [1,1,1,1,1]
    params, covariance = curve_fit(polynomial, x_continuum, y_continuum, p0=initial_guess, sigma=y_err, absolute_sigma=True)
    fit = lambda x: polynomial(x,*params)

    return fit, x_continuum, y_continuum

## filtering out peaks ##

def filterout_peaks(x,y):
    Q1 = np.percentile(y, 30)
    Q3 = np.percentile(y, 70)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    mask= y<threshold
    filtered_x=(x)[mask]
    filtered_y=(y)[mask]
    #filtered_y = y[y < threshold]  # Remove peaks
    
    return filtered_x, filtered_y


## EW ##

def EW_parametric(x,y,cont,method=0,plots=True): #


    max=np.argmax(y)
    bound1,bound2=x[max]-10,x[max]+10

    if plots==True:
        plt.figure(figsize=(8, 5))
    
    if method==0: # computing the EW from a fit to (flux-continuum)
        
        x,y=chop_data(x,y,bound1,bound2)
        # removing the continuum from the flux data
        flux_reduced = cont(x)-y
        
        
        y_err=np.full(len(flux_reduced), mad(flux_reduced))#np.sqrt(np.median(ebulge, axis=(1, 2)));
        if plots==True:
            plt.fill_between(x,flux_reduced - y_err, flux_reduced + y_err, color='blue', alpha=0.1, label="Uncertainty")

        initial_guess = [np.max(flux_reduced), np.mean(x), np.std(x)]
        params, covariance = curve_fit(gaussian, x, flux_reduced, p0=initial_guess, sigma=y_err, absolute_sigma=True)
        
        #print("cov ",covariance)
        #print("sigma f is ", gaussian_error(x, params, covariance))

        print(" ")

        A_fit, mu_fit, sigma_fit = params

        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = gaussian(x_fit, params[0], params[1], params[2])

        flux_reduced_gaussian = lambda x: gaussian(x,*params)
        
        xx=np.linspace(bound1,bound2, 100)  # Generate 100 new points
        
        delta=xx[2]-xx[1]
        
        val=0
        err=0
        for xi in xx:
            val+=delta*(flux_reduced_gaussian(xi))/cont(xi)
            #err+=delta*(np.sqrt())

        
    else: # computing the EW from a fit to flux (continuum+gaussian)

        y_err=np.full(len(y), mad(y))#np.sqrt(np.median(ebulge, axis=(1, 2)));
        if plots==True:
            plt.fill_between(x,y - y_err, y + y_err, color='blue', alpha=0.4, label="Uncertainty",zorder=4)


        #c0, c1, c2, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3
        initial_guess = [0.01, 10, 0.05, np.max(y) , np.mean(x), 5, np.max(y)/3, np.mean(x)-15.5, 3, np.max(y)/3 , np.mean(x)+21, 5]
        params, covariance = curve_fit(three_gaussian_poly, x, y, p0=initial_guess, maxfev=5000, sigma=y_err, absolute_sigma=True)
        c0_fit, c1_fit, c2_fit, A1_fit, mu1_fit, sigma1_fit, A2_fit, mu2_fit, sigma2_fit, A3_fit, mu3_fit, sigma3_fit = params
                
        x_fit = np.linspace(np.min(x), np.max(x), 500)
        y_fit = three_gaussian_poly(x_fit, *params)

        
        # flux_reduced_gaussian is (-1)*gaussian of the main peak
        flux_reduced_gaussian = lambda x: -gaussian(x,params[3],params[4],params[5])
        flux_reduced = flux_reduced_gaussian(x)


        cont = lambda x: three_gaussian_poly(x,*params) + flux_reduced_gaussian(x)
        
        xx=np.linspace(bound1,bound2, 100)  # Generate 100 new points
        
        delta=xx[2]-xx[1]
        
        val=0
 
        for xi in xx:
            val+=delta*(flux_reduced_gaussian(xi))/cont(xi)


    
    if plots==True:
    
        #plot of error propagated to f
        #plt.fill_between(x, cont(x)-y-gaussian_error(x, params, covariance),cont(x)-y+gaussian_error(x, params, covariance), alpha=0.3, color='red', label="propagated error of f")
        
        plt.scatter(x, y, label="Original data", color="black",s=10)
        plt.plot(x,cont(x), label="Continuuum")
        plt.scatter(x, flux_reduced, label="Continuum-Flux", color="red",s=3)
        plt.plot(x_fit, y_fit, label="fit")
        plt.fill_between(x_fit, y_fit, alpha=0.3, color='gray', label="Integral")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.legend()
        plt.title("Finding EW")
        plt.show()


    return val,err
    
    




## Maps ##
#parametric#
def EW_map_parametric(cube_region,wave,central_wavelength,kernel_size=3,method=0, plots=False):
        

    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    map=np.zeros((x_len, y_len))
    error_map=np.zeros((x_len, y_len))
    
    x_min=central_wavelength-40
    x_max=central_wavelength+40
            
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]
            
            # chop data
            x_chopped,y_chopped=chop_data(wave,spec,x_min,x_max)
            
            
            # smooth data
            y_smooth=smooth_spectra(y_chopped,kernel_size)
            
            # fit to continuum
            aux=continuum(x_chopped,y_smooth)
            if(aux==None): # skipping the i,j pixel in case we cannot find the continuum
                map[i,j]=np.nan
                continue 
            else:
                continuum_fit=aux[0]
                x_cont=aux[1]
                y_cont=aux[2]
            
            y_continuum_fit = continuum_fit(x_chopped)            

            # measuring EW
            """if i==21 and j==15:
                plots=True"""
            ew,err=EW_parametric(x_chopped,y_smooth,continuum_fit,method, plots)
            
            print("EW=",ew," at (i,j)=",i,",",j)
            
            map[i,j]=ew
            error_map[i,j]=err
            
    return map, error_map

def EW_map_non_parametric(cube_region,wave,central_wavelength,kernel_size=3,plots=False):

    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    map=np.zeros((x_len, y_len))
    
    
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]
            
            x_chopped,y_chopped=chop_data(wave,spec,central_wavelength-50,central_wavelength+50)

            # smooth data
            y_smooth=smooth_spectra(y_chopped,kernel_size) # this is not doing anything


            # compute continuum with kernel
            x,y=filterout_peaks(x_chopped,y_smooth)
            kernel = cosine_kernel(3)
            cont = convolve(y, kernel, mode='same')
            
            max=np.argmax(y_smooth)
            b1=x_chopped[max]-10
            b2=x_chopped[max]+10
            x,y=chop_data(x_chopped,y_smooth,b1,b2)

            x_cont,y_cont=filterout_peaks(x_chopped,y_smooth)

            kernel = cosine_kernel(4)
            cont = convolve(y_cont, kernel, mode='same')
            interp=interp1d(x_cont, cont, kind='cubic')


            
            continuum = interp(x)

            
            # Compute the excess intensity above the continuum            
            excess_intensity = (continuum-y)/continuum

            # Integrate the excess intensity (area over the continuum)
            #area_over_continuum = trapz(excess_intensity, x)
            
            # Interpolating before integrating, so that we have more points
            f = interp1d(x, excess_intensity, kind='cubic')

            xx=np.linspace(np.min(x),np.max(x), 100)  # Generate 100 new points
            excess_intensity = f(xx)


            # Integrate the excess intensity (area over the continuum)
            area_over_continuum = trapz(excess_intensity, xx)


            

            """if i==21 and j==15:
                plots=True"""


            if plots==True:
                plt.plot(x, continuum-y, label="cont-spec", color="blue")
                plt.plot(x, continuum, label="Continuum", linestyle="dashed", color="red")
                plt.fill_between(x, continuum-y, 0, alpha=0.3, color="green", label="Excess area")
                plt.xlabel("Wavelength")
                plt.ylabel("Intensity")
                plt.legend()
                plt.show()

            
            ew=area_over_continuum;
            print(f"Integral of area over continuum divided by continuum: {area_over_continuum:.2f} at (i,j)=",i,", ",j)
            
            map[i,j]=ew
    return map



def velocity(x,y,cont,lambda_rest):

    #xx=np.linspace(np.min(x),np.max(x), 100)  # Generate 100 new points

    c = 3*10**5 # in km/s


    num,denom=0,0
    for i in range(len(x)):
        denom += 1-y[i]/cont(x[i])
        num +=  (1-y[i]/cont(x[i])) * x[i]

    v = c/lambda_rest * (num/denom - lambda_rest)

    return v

def velocity_map(cube_region,wave,lambda_obs,lambda_rest,kernel_size=3):

    x_len=len(cube_region[0][0])
    y_len=len(cube_region[0])
    
    map=np.zeros((x_len, y_len))
    
    
    for i in range(0,x_len):
        for j in range(0,y_len):
            
            spec=cube_region[:,j,i]

            # chop data
            x_chopped,y_chopped=chop_data(wave,spec,lambda_obs-40,lambda_obs+40)
                        
            # smooth data
            y_smooth=smooth_spectra(y_chopped,3)
                        
            # fit to continuum
            cont=continuum(x_chopped,y_smooth)[0]
                        
            
            map[i,j]=velocity(x_chopped,y_chopped,cont,lambda_rest)

    return map