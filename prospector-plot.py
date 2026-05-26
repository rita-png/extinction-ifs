import os
os.environ['SPS_HOME'] = '/home/ritapsantos/PhD/fsps'
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("output_dynesty.pkl", "rb") as f:
    output_dynesty = pickle.load(f)

sampling = output_dynesty["sampling"]

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
plt.savefig("corner.png", dpi=150)