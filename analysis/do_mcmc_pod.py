#region Imports
import pandas as pd
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

from lqg.tracking.basic import TrackingTask

import multiprocessing as mp  # Needed for CPU count

# Force JAX to run on CPU instead of GPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Get available CPU cores (should return 32)
cpu_count = mp.cpu_count()
print(f"Using {cpu_count} CPU cores.")

# Set NumPyro to fully use all CPU threads
numpyro.set_host_device_count(cpu_count)

#endregion

#region Utility functions
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Filter lines that contain valid CSV rows (assuming at least 9 columns, adjust if needed)
    valid_lines = [line for line in lines if line.count(",") >= 4]  # Adjust based on actual column count
    
    # Read the filtered lines into a DataFrame
    data = pd.read_csv(io.StringIO("".join(valid_lines)))

    # Extract relevant columns
    relevant_columns = ["Block", "Tracked_x", "Tracked_y", "Mouse_x", "Mouse_y"]
    data = data[relevant_columns]

    # Convert to dictionary: {block: [[Tracked_x, Tracked_y, Mouse_x, Mouse_y], ...]}
    block_dict = {}
    for _, row in data.iterrows():
        block = row["Block"]
        values = row[1:].tolist()  # Strip 'Block' column
        if block not in block_dict:
            block_dict[block] = []
        block_dict[block].append(values)

    return {k: jnp.array(v) for k, v in block_dict.items()}
#endregion

#region LQG functions
def lqg_model(x):
    # priors
    action_variability = numpyro.sample("action_variability", dist.HalfCauchy(1.))
    action_cost = numpyro.sample("action_cost", dist.HalfCauchy(1.))
    sigma_target = numpyro.sample("sigma_target", dist.HalfCauchy(50.))
    sigma_cursor = numpyro.sample("sigma_cursor", dist.HalfCauchy(15.))

    model = TrackingTask(
        dim = 2,
        action_variability = action_variability,
        action_cost = action_cost,
        sigma_target = sigma_target,
        sigma_cursor = sigma_cursor,
        dt = 1. / 60,
        T = (x.shape)[1]
    )

    # likelihood
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])
#endregion

data_path = "C:/Users/Beno/Documents/CEU/continuous_psychophysics/vsl_with_tracking/outputs"
data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "outputs"))
#region Main execution setup
if __name__ == "__main__":
    # Configuration moved to main block
    # Detect CPU count first
    cpu_count = mp.cpu_count()

    # Configure JAX
    jax.config.update("jax_platform_name", "cpu")  # Force CPU
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

    # Optimize thread parallelism dynamically
    os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cpu_count}"

    print(f"Using {cpu_count} CPU cores.")

    # Define worker initializer for parallel processes
    def init_worker():
        #numpyro.set_host_device_count(8)
        pass

    # Modify MCMC function to use subject-specific random seed
    def do_mcmc_for_one_subject(series, subject):
        subject_path = f"{data_path}/{series}/{subject}"
        data_by_blocks = load_data(f"{subject_path}/tracking.txt")
        data = jnp.array(
            [chunk for v in data_by_blocks.values() for chunk in jnp.split(v[: (len(v) // 1000) * 1000], len(v) // 1000)]
        ).reshape(-1, 1000, 4)

        print(f"Start subject {subject}, data shape: {data.shape}.")
        nuts_kernel = NUTS(lqg_model)
        mcmc = MCMC(nuts_kernel, num_warmup=250, num_samples=500, num_chains=2)
        
        # Use subject-specific PRNGKey
        mcmc.run(random.PRNGKey(subject), data)  # Changed seed to use subject ID
        
        mcmc_results = az.from_numpyro(mcmc)
        mcmc_results.to_netcdf(f"{subject_path}/mcmc_results_2.nc")
        print(f'MCMC results saved for subject {subject}.')

    # Create parallel tasks and execute
    with mp.Pool(processes=cpu_count, initializer=init_worker) as pool:
        # Create arguments list: [(series=1, subject=7), (1,8), ..., (1,15)]
        tasks = [(1, subject) for subject in range(7, 16)]
        pool.starmap(do_mcmc_for_one_subject, tasks)

    print("All subjects processed in parallel.")