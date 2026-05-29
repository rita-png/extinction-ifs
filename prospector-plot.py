import os
os.environ['SPS_HOME'] = '/home/ritapsantos/PhD/fsps'
import numpy as np
import matplotlib.pyplot as plt
import pickle

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("output_dynesty.pkl", "rb") as f:
    output = pickle.load(f)

sampler = output["sampling"]

chain = sampler.get_chain()
flat_chain = sampler.get_chain(flat=True)
log_prob = sampler.get_log_prob(flat=True)

ndim = flat_chain.shape[1]
print("ndim:", ndim)

#get the median values of the parameters
"""theta_med = np.median(flat_chain, axis=0)
print(theta_med)
print(output.keys())
model = output["model"]
obs   = output["obs"]
sps   = output["sps"]
spec, phot, mfrac = model.predict(theta_med, obs=obs, sps=sps)

plt.figure(figsize=(10,5))

# plot observed spectrum
if obs["wavelength"] is not None:
    plt.plot(obs["wavelength"], obs["spectrum"], label="Observed", alpha=0.7)

# plot modelled spectrum
plt.plot(obs["wavelength"], spec, label="Median model", lw=2)

plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.legend()
plt.tight_layout()
plt.savefig("median_spectrum.png", dpi=150)
plt.show()"""


# adjust this list to match ndim
theta_labels =  theta_labels = ["mass", "logzsol", "dust2", "tage", "tau", "dust_index","sigma_smooth"] #"sigma_smooth","dust_index"]
print("labels:", theta_labels)

# trace plot
fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim))
for i, (ax, label) in enumerate(zip(axes, theta_labels)):
    ax.plot(chain[:, :, i], alpha=0.3)
    ax.set_ylabel(label)
axes[-1].set_xlabel("iteration")
plt.tight_layout()
plt.savefig("trace_emcee.png", dpi=150)

# corner plot
from prospect.plotting import corner
cfig, caxes = plt.subplots(ndim, ndim, figsize=(10,9))
caxes = corner.allcorner(flat_chain.T, theta_labels, caxes, color="royalblue", show_titles=True)
plt.savefig("corner_emcee.png", dpi=150)


#DYNESTY IS BELOW
"""

with open("output_dynesty.pkl", "rb") as f:
    output_dynesty = pickle.load(f)

sampling = output_dynesty["sampling"]
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullFormatter

print("keys:", sampling.keys())
print("points shape:", sampling["points"].shape)
print("log_weight shape:", sampling["log_weight"].shape)

# the chain is the points array
chain = sampling["points"]
weights = np.exp(sampling["log_weight"] - sampling["log_weight"].max())
weights /= weights.sum()

# i have to be careful with the order of the parameters, but here i know it from the model definition
theta_labels = ["mass", "logzsol", "dust2", "tage", "tau", "sigma_smooth", "dust_index"]

# trace plot
fig, axes = plt.subplots(chain.shape[1], 1, figsize=(10, 2*chain.shape[1]))
for i, (ax, label) in enumerate(zip(axes, theta_labels)):
    ax.plot(chain[:, i], alpha=0.5)
    ax.set_ylabel(label)
axes[-1].set_xlabel("iteration")
plt.tight_layout()
plt.savefig("trace.png", dpi=150)

# corner plot
from prospect.plotting import corner
from prospect.plotting.utils import best_sample
nsamples, ndim = chain.shape
cfig, axes = plt.subplots(ndim, ndim, figsize=(10,9))
axes = corner.allcorner(chain.T, theta_labels, axes, weights=weights, color="royalblue", show_titles=True)
for ax in axes[:,0]:
    ax.set_xscale('log')
    

plt.tight_layout()

plt.savefig("corner.png", dpi=150)"""