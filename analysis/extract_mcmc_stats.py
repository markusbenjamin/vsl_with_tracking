from scipy.stats import mode
import arviz as az
import json
import numpy as np
import os

raw_path = os.path.abspath(os.path.join(os.getcwd(), "..", "outputs"))
data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "outputs"))

def extract_mcmc_summary(series, subject, run):
    subject_path = os.path.join(os.getcwd(), "data", str(series), str(subject))
    inference_data = az.from_netcdf(f"{subject_path}/mcmc_results_{run}.nc")

    posterior = inference_data.posterior

    # Compute MAP estimates (mode along draws)
    map_estimates = {
        var: mode(posterior[var].values, axis=1, keepdims=False).mode.squeeze().tolist()
        for var in posterior.data_vars
    }

    print(f'{subject}: {map_estimates}')
    
    if True:
        # Extract clean R-hat values
        raw_r_hat = az.rhat(inference_data).to_dict()
        r_hat_clean = {param: raw_r_hat["data_vars"][param]["data"] for param in raw_r_hat["data_vars"]}

        # Store everything in a single nested dict
        mcmc_summary = {
            "MAP_estimates": map_estimates,
            "R_hat": r_hat_clean  # Now it's a clean dictionary
        }

        json.dump(mcmc_summary, open(f"{subject_path}/mcmc_summary_{run}.json", "w"), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

series = 1
run = "4"
for subject in range(1, 2):
    extract_mcmc_summary(series, subject,run)