import h5py
import numpy as np
from scipy.stats import truncnorm

import jax
from numpyro.infer import MCMC, NUTS
import arviz as az

from intensity_model import samples_model

input_file = "q_obs.h5"
output_file = "inference_result_200.h5" # "inference_result.h5"

dataset_name = "q_obs"

lower = 0.0
upper = 1.0
sigma_obs = 0.1
Nsamp = 200 # 1000
nmcmc = 1000
nchain = 1
seed = 123

with h5py.File(input_file, "r") as hf:
    q_obs = np.array(hf[dataset_name])

q_obs = np.asarray(q_obs, dtype=float)
Nobs = q_obs.shape[0]

print(f"Nobs = {Nobs}")

rng = np.random.default_rng(seed)

a = (lower - q_obs) / sigma_obs
b = (upper - q_obs) / sigma_obs

q_samples = truncnorm.rvs(a[:, None], b[:, None], loc=q_obs[:, None], scale=sigma_obs, size=(Nobs, Nsamp), random_state=rng,).astype(np.float32)

print(f"q_samples shape = {q_samples.shape}")

kernel = NUTS(samples_model)
mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain,)

mcmc.run(jax.random.PRNGKey(seed), q_samples, lower=lower, upper=upper,)

trace = az.from_numpyro(mcmc)

mu_samples = np.array(trace.posterior.mu).flatten()
sigma_samples = np.array(trace.posterior.sigma).flatten()

neff_samples = None
if hasattr(trace.posterior, "neff"):
    neff_samples = np.array(trace.posterior.neff)

with h5py.File(output_file, "w") as hf:
    hf.create_dataset("q_obs", data=q_obs)
    hf.create_dataset("q_samples", data=q_samples)
    hf.create_dataset("mu", data=mu_samples)
    hf.create_dataset("sigma", data=sigma_samples)

    if neff_samples is not None:
        hf.create_dataset("neff", data=neff_samples)

    hf.attrs["lower"] = lower
    hf.attrs["upper"] = upper
    hf.attrs["sigma_obs"] = sigma_obs
    hf.attrs["Nsamp"] = Nsamp
    hf.attrs["num_warmup"] = nmcmc
    hf.attrs["num_samples"] = nmcmc
    hf.attrs["num_chains"] = nchain
    hf.attrs["seed"] = seed

print(f"Saved inference results to: {output_file}")