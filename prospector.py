#import prospect
"""
confirmar se é preciso corrigir o redshift

param de resolucao do espetro deixar livre

200 ou 500 interacoes no dynasty

redshfit livre

mask emission lines"""






from functions import *

#use 1/10 galaxy mass OR use full integrated light
#take a look at the manual

# to do:
# 
# use photometry
#     
#     inspect what choices i can do in the model (w/ w/out dust idk)
#     
#     install cigale
#     
#     compare what params cigale and prospector outputs


#binnign in wavelenght! ttry, give uncertainties, use photometry


#pick the galaxy accordign to whatever i have data from cosmos2020 and also MUSE


import os
os.environ['SPS_HOME'] = '../fsps'

#os.environ['SPS_HOME'] = '/home/rita13santos/PhD/fsps'
import prospect
print(prospect.__version__)

import fsps
import dynesty
import sedpy
import h5py, astropy
import numpy as np
import astroquery


z=0.009213#redshift of SN2010ev host (NGC3244)



# import prospector packages
from sedpy.observate import load_filters
#from prospect.utils.obsutils import fix_obs
from prospect.observation.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
#from prospect.models.sedmodel import SedModel
from prospect.models.sedmodel import SpecModel as SedModel
from prospect.models import priors
from prospect.sources import CSPSpecBasis #use FastStepBasis for parametric SFH


# Importing data

#from prospect.fitting import fit_model;
#help(fit_model)
import astropy.io.fits as fits

#import data
file_name="../DATA/SN2001el.fits"#SN2010ev.fits" 
data = fits.open(file_name)
cube = data[1].data   # this is the cube, a (3681 x 341 x 604) matrix with fluxes at different 3681 wavelengths and 308 x 318 spatial pixels ("spaxels")
header = data[1].header # this has information on the data cube
ecube = data[2].data # this is the cube uncertainty (3681 x 341 x 604)
print(np.shape(cube))


print(cube[0])
print(cube[0][200][300])

x_len=len(cube[0][0])
y_len=len(cube[0])

#  -- following is to the get the wavelength array
CRVAL = float(header["CRVAL3"])
NAXIS = int(header["NAXIS3"])
CDELT = float(header["CD3_3"])
CRPIX = float(header["CRPIX3"])
wave = np.array(CRVAL + CDELT * (np.arange(NAXIS) - CRPIX))

print("Raw flux range:", np.nanmin(cube[0]), np.nanmax(cube[0]))
print("Header BUNIT:", header.get('BUNIT', 'not found'))



x_center, y_center = int(x_len/2), int(y_len/2)
x_half, y_half = 5,5 #300, 300

x_min = x_center - x_half
x_max = x_center + x_half
y_min = y_center - y_half
y_max = y_center + y_half


region = cube[:, y_min:y_max, x_min:x_max]
region_err = ecube[:, y_min:y_max, x_min:x_max]

#flux = np.nansum(region, axis=(1, 2))/(len(region[0][0])*len(region[0]))
#err = np.nansum(region_err, axis=(1, 2))/(len(region[0][0])*len(region[0]))

# new flux and uncertainties in maggies!
flux_sum = np.nansum(region, axis=(1, 2)) / (len(region[0][0]) * len(region[0]))
err_sum = np.sqrt(np.nansum(region_err**2, axis=(1, 2))) / (len(region[0][0]) * len(region[0]))

flux = (flux_sum / 1e20) * wave**2 * 3.34e4 / 3631.0
err  = (err_sum  / 1e20) * wave**2 * 3.34e4 / 3631.0
err = np.clip(err, a_min=0.1 * np.nanmedian(flux), a_max=None) #NOVO


print("Change spectra uncertainty above") #use the one from my other code file, milky-way.py


# build observation in new prospector version
from prospect.observation import Spectrum

obs = Spectrum(
    wavelength=np.array(wave, dtype=np.float64),
    spectrum=np.array(flux, dtype=np.float64),
    unc=np.array(err, dtype=np.float64),
    redshift=z,
    mask=np.isfinite(flux)
)

    
observations = [obs]

print("Clip threshold:", 0.15 * np.nanmedian(obs['spectrum']))
print("Unc min:", np.nanmin(obs['unc']))

# set model

from prospect.models.templates import TemplateLibrary

# new priors #

model_params = TemplateLibrary["parametric_sfh"]# ['continuity_sfh']
model_params.update(TemplateLibrary["spectral_smoothing"])

model_params["zred"]["init"] = z
model_params["zred"]["isfree"] = False

model_params["mass"]    = {'N': 1, 'isfree': True, 'init': 3e4, 'prior': priors.LogUniform(mini=1e4, maxi=1e8)}
model_params["logzsol"] = {'N': 1, 'isfree': True, 'init': -0.5, 'prior': priors.TopHat(mini=-2.0, maxi=0.19)}
model_params["dust2"]   = {'N': 1, 'isfree': True, 'init': 0.1,  'prior': priors.TopHat(mini=0.0, maxi=4.0)}
model_params["tage"]    = {'N': 1, 'isfree': True, 'init': 1.0,  'prior': priors.TopHat(mini=0.1, maxi=13.8)}
model_params["tau"]     = {'N': 1, 'isfree': True, 'init': 1.0,  'prior': priors.LogUniform(mini=0.1, maxi=30)}
model_params["dust_type"]  = {'N': 1, 'isfree': False, 'init': 4} # Calzetti # Kriek & Conroy, allows for free R_V
model_params["dust_index"] = {'N': 1, 'isfree': True, 'init': -0.7, 'prior': priors.TopHat(mini=-3.0, maxi=0.4)}

#

"""
mass,logzsol,dust2,rv =  9.5,-0.5,0.5,3.1

model_params["logmass"]["init"] = mass
model_params["dust2"]["init"] = dust2

# age bins in years
bins_yr = [
    [1e6, 3e7],
    [3e7, 1e8],
    [1e8, 3e8],
    [3e8, 1e9],
    [1e9, 3e9],
    [3e9, 1e10],
]
nbins_sfh = len(bins_yr)
agebins = np.log10(bins_yr)

model_params["agebins"] = {'N': nbins_sfh,'isfree': False,'init': agebins,'units': 'log(yr)'}

model_params["logsfr_ratios"] = {
    'N': nbins_sfh - 1,
    'isfree': True,
    'init': np.zeros(nbins_sfh - 1),
    'prior': priors.StudentT(mean=np.zeros(nbins_sfh - 1),
                             scale=np.ones(nbins_sfh - 1) * 0.3,
                             df=np.ones(nbins_sfh - 1) * 2)
}

#creating a SSP
#Simple Stellar Population (SSP) is a group of stars born at the same time from the same gas cloud,


#print(sps.ssp.libraries)

"""
sps = CSPSpecBasis(zcontinuous=1)#CSPSpecBasis(zcontinuous=1)
model = SedModel(model_params) #initializing the model based on the parameters we defined
print("The model is\n\n", model)


########
print("######")
#mass: Total stellar mass formed (in solar masses)
#logzsol: Logarithmic metallicity
#dust2:  V-band dust attenuation (like A_V)
#tage: Age of the galaxy (time since star formation began)
#tau: Star formation timescale (for parametric SFHs, like τ-model)
#dust_index:Slope of the dust curve (like R_V; only used with dust_type=2)



noise_model = (None, None) #i HAVE TO CHANGE THIS, A tuple of NoiseModel objects for the spectroscopy and photometry respectively. Can also be (None, None) in which case simple chi-square will be used




#test
spec, mfrac = model.predict(model.theta, observations=observations, sps=sps)

print("Data range:  ", np.nanmin(obs['spectrum']), np.nanmax(obs['spectrum']))
print("Model range: ", np.nanmin(spec), np.nanmax(spec))
print("Unc range:   ", np.nanmin(obs['unc']), np.nanmax(obs['unc']))


print("IMPORTANT\n\nData max:  ", np.nanmax(obs.flux))
print("Model max: ", np.nanmax(spec[0]))
print("Ratio:     ", np.nanmax(obs.flux) / np.nanmax(spec[0]))


plt.plot(obs.wavelength, obs.flux, label='data')
plt.plot(obs.wavelength, spec[0], label='model')
plt.legend()
plt.savefig('best_fit_optimize.png', dpi=150, bbox_inches='tight')
plt.show(block=True)  # block=True forces the script to wait
print("N wavelength points:", len(obs['wavelength']))

#

# make a prediction
print("\nParameter values used to make a prediction model are ", model.theta)


#generate synthetic photometry and spectroscopy (a Model SED) for a given set of stellar population parameters

current_parameters = ",".join([f"{p}={v}" for p, v in zip(model.free_params, model.theta)])


"""
plt.title("Synthetic spectra (for the stellar population) produced from the input parameters")
plt.plot(wave, spec)
plt.show()
"""

print("Inspect this, these are the model bounds for the parameters, if they are too wide the fit will not converge to a given solution: ")

for name, bounds in zip(model.free_params, model.theta_bounds()):
    print(name, bounds)





# fit
print("Fitting now")

from prospect.fitting import lnprobfn, fit_model

fitting_kwargs = dict(nlive_init=100, nested_method="rwalk", nested_target_n_effective=100, nested_dlogz_init=2) #400, 100, 0.05

output = fit_model(observations, model, sps, noise=noise_model, dynesty=False, optimize=True, **fitting_kwargs,verbose=True)
# for optimize==True, results are here:
result = output["optimization"]


theta_best = output["optimization"][0][0].x
print("Best fit parameters from optimization:")
for name, val in zip(model.free_params, theta_best):
    print(f"  {name}: {val}")



# plot best fit (frooptimize==True)
spec, mfrac = model.predict(theta_best, observations=observations, sps=sps)

plt.figure()
plt.plot(obs.wavelength, obs.flux, label='data')
plt.plot(obs.wavelength, spec[0], label='best fit')  # spec is now a list
plt.legend()
plt.show()


print("\nUsing these best fit parameters as initial values for dynesty sampling, to get the posterior distribution of the parameters\n")
print("###### dynesty fitting ######")
# set initial values for dynesty from optimze fit
for i, (name, val) in enumerate(zip(model.free_params, theta_best)):
    model_params[name]["init"] = val

# reinitialize model with new inits
model = SedModel(model_params)
fitting_kwargs = dict(nlive_init=200, nested_method="rwalk", nested_target_n_effective=200, nested_dlogz_init=0.5)

output_dynesty = fit_model(observations, model, sps, noise=noise_model, nested_sampler='dynesty', optimize=False, **fitting_kwargs, verbose=True)



print(output_dynesty["sampling"])



result, duration = output_dynesty["sampling"]

# save results
from prospect.io import write_results as writer
hfile = "./quickstart_dynesty_mcmc.h5"
writer.write_hdf5(hfile, {}, model, observations,
                 output_dynesty["sampling"][0], None,
                 sps=sps,
                 tsample=output_dynesty["sampling"][1],
                 toptimize=0.0)




# plotting
from prospect.io import read_results as reader
hfile = "./quickstart_dynesty_mcmc.h5"
out, out_obs, out_model = reader.results_from(hfile)


import matplotlib.pyplot as pl
from prospect.plotting import corner
nsamples, ndim = out["chain"].shape
cfig, axes = pl.subplots(ndim, ndim, figsize=(10,9))
axes = corner.allcorner(out["chain"].T, out["theta_labels"], axes, weights=out["weights"], color="royalblue", show_titles=True)

from prospect.plotting.utils import best_sample
pbest = best_sample(out)
corner.scatter(pbest[:, None], axes, color="firebrick", marker="o")

pl.show()

from prospect.plotting.utils import best_sample
theta_best = best_sample(out)
spec, mfrac = model.predict(theta_best, observations=observations, sps=sps)
plt.plot(obs.wavelength, obs.flux, label='data')
plt.plot(obs.wavelength, spec[0], label='best fit')

# total stellar mass formed (in solar masses), age of the galaxy, metallicity, diffuse dust attenuation (Av)


# # CIGALE

"""
graphs
parametric or non parametric SFH? (i am doing non parametric)
get_ipython().set_next_input('what values should i use for my prior');get_ipython().run_line_magic('pinfo', 'prior')
"""
