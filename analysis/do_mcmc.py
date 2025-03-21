#region Imports
import csv
import linalg
import io
import os
import jax.numpy as jnp
from jax.random import PRNGKey, split
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

import jax
from jax import random, numpy as jnp

import numpyro
from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC
import arviz as az

from lqg.tracking.basic import TrackingTask, BoundedTrackingTask

import multiprocessing as mp  # Needed for CPU count

# Configure JAX to use all available CPU cores
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(12)

#jax.config.update("jax_platform_name", "gpu")  # Use GPU instead of CPU
#jax.config.update("jax_enable_x64", True)

#device_count = jax.device_count()  # Auto-detect GPUs
#cpu_count = mp.cpu_count()  # Detect available CPUs
#
#numpyro.set_host_device_count(device_count if jax.devices()[0].device_kind == "gpu" else cpu_count)

#endregion

#region Utility functions
def load_data(file_path):
    block_dict = {}
    required_keys = {"Block", "Tracked_x", "Tracked_y", "Mouse_x", "Mouse_y"}
    
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Check if row contains all required keys
            if not required_keys.issubset(row.keys()):
                continue
            
            block = row["Block"]
            try:
                # Convert values to floats (or ints, if appropriate)
                target_x = float(row["Tracked_x"])
                target_y = float(row["Tracked_y"])
                tracking_x = float(row["Mouse_x"])
                tracking_y = float(row["Mouse_y"])
            except ValueError:
                # Skip rows with non-numeric data
                continue
            
            # Rearrange the order to [target_x, tracking_x, target_y, tracking_y]
            values = [target_x, tracking_x, target_y, tracking_y]
            
            # Group by block
            if block not in block_dict:
                block_dict[block] = []
            block_dict[block].append(values)
    
    # Convert lists to JAX arrays
    return {k: jnp.array(v) for k, v in block_dict.items()}
#endregion

#region LQG functions
def lqg_model(x):
    # priors
    #action_variability = numpyro.sample("action_variability", dist.HalfCauchy(1.)) #sigma_tracking
    action_variability = numpyro.sample("action_variability",dist.LogNormal(2.06497, 0.396418))
    action_cost = numpyro.sample("action_cost", dist.HalfCauchy(14.)) #action_cost
    sigma_target = numpyro.sample("sigma_target", dist.HalfCauchy(700.)) #psi_target
    sigma_cursor = numpyro.sample("sigma_cursor", dist.HalfCauchy(250.)) #psi_tracking

    model = TrackingTask(
        dim = 2,
        process_noise=14.7, #sigma_target
        action_variability = action_variability,
        action_cost = action_cost, # experiment with this
        sigma_target = sigma_target,
        sigma_cursor = sigma_cursor,
        dt =  1. / 60,#3. / 50,
        T = (x.shape)[1]
    )

    # likelihood
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])

def bounded_lqg_model(x):
    # Priors for model parameters
    action_variability = numpyro.sample("action_variability", dist.HalfCauchy(1.))
    action_cost = numpyro.sample("action_cost", dist.HalfCauchy(1.))
    sigma_target = numpyro.sample("sigma_target", dist.HalfCauchy(50.))
    sigma_cursor = numpyro.sample("sigma_cursor", dist.HalfCauchy(15.))
    
    # Instantiate BoundedTrackingTask with explicit boundaries
    model = BoundedTrackingTask(
        dim=2,
        target_min=[452, 1476],    # Explicit boundaries for each dimension
        target_max=[24, 1044],
        action_variability=action_variability,
        action_cost=action_cost,
        sigma_target=sigma_target,
        sigma_cursor=sigma_cursor,
        dt=1./60,
        T=x.shape[1]
    )
    
    # Likelihood: use the modified conditional_distribution for reflective transitions
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])
#endregion

raw_path = os.path.abspath(os.path.join(os.getcwd(), "..", "outputs"))
data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "outputs"))

def do_mcmc_for_one_subject(series, subject, run):
    subject_path = os.path.join(os.getcwd(), "data", str(series), str(subject))

    data_by_blocks = load_data(f"{subject_path}/tracking.txt")

    data = jnp.array(
        [chunk for v in data_by_blocks.values() for chunk in jnp.split(v[: (len(v) // 1000) * 1000], len(v) // 1000)]
    ).reshape(-1, 1000, 4)  # Ensure the correct shape

    print(f"Start subject {subject}, data shape: {data.shape}.")

    #nuts_kernel = NUTS(lqg_model)
    nuts_kernel = NUTS(lqg_model)

    mcmc = MCMC(nuts_kernel, num_warmup=250, num_samples=500, num_chains=2)
    mcmc.run(random.PRNGKey(0), data)

    print(f'MCMC finished for subject {subject}.')

    mcmc_results = az.from_numpyro(mcmc)
    mcmc_results.to_netcdf(f"{subject_path}/mcmc_results_{run}.nc")

    print(f'MCMC results saved for subject {subject}.')

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")

run = "4"
series = 1
for subject in range(1, 2):
    do_mcmc_for_one_subject(series, subject, run)

"""

if False:
    x_target = data[:,0]
    y_target = data[:,1]
    x_tracking = data[:,2]
    y_tracking = data[:,3]

    plt.plot(x_target, y_target, label="Target Path (x, y)")
    plt.plot(x_tracking, y_tracking, label="Tracking Path (x, y)")
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("2D Tracking and Target Data")
    plt.show()

if False:
    vel_x_target = jnp.array([jnp.diff(x_target)])
    vel_y_target = jnp.array([jnp.diff(y_target)])
    vel_x_tracking = jnp.array([jnp.diff(x_tracking)])
    vel_y_tracking = jnp.array([jnp.diff(y_tracking)])

    x_lags, x_correls = xcorr(vel_x_tracking, vel_x_target, maxlags=120)
    y_lags, y_correls = xcorr(vel_y_tracking, vel_y_target, maxlags=120)

    plt.plot(x_lags, x_correls.mean(axis=0), label="X coordinate")
    plt.plot(y_lags, y_correls.mean(axis=0), label="Y coordinate")
    plt.xlabel("Lag [s]")
    plt.ylabel("Cross-correlation")
    plt.title("2D cross-correlogram")
    plt.show()

"""