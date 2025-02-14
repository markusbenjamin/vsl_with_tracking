from scipy.signal import fftconvolve
import arviz as az
import json
import numpy as np
import jax.numpy as jnp
import pandas as pd
import io

data_path = "C:/Users/Beno/Documents/CEU/continuous_psychophysics/vsl_with_tracking/outputs"

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

def xcorr(x, y, maxlags=60, normed=True):
    """ Compute cross correlation between two arrays along last axis of an array,
    treating other axes as batch dimensions

    Args:
        x, y: arrays of same dimensions
        maxlags: cutoff
        normed: normalize the correlations

    Returns:
        lags, correlations
    """
    Nx = x.shape[-1]

    correls = fftconvolve(x, y[..., ::-1], mode="full", axes=-1)

    if normed:
        eps = 1e-10  # Small constant to prevent log(0) and div by zero
        log_x = np.log(np.sum(x * x, axis=-1) + eps)
        log_y = np.log(np.sum(y * y, axis=-1) + eps)

        denom = np.exp(0.5 * (log_x + log_y))  # Log-sum-exp trick for stability
        correls = correls / np.maximum(eps, denom)[..., None]  # Ensure stability in division

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)

    correls = correls[..., Nx - 1 - maxlags:Nx + maxlags]
    return lags, correls

def do_xcorr(series, subject):
    subject_path = f"{data_path}/{series}/{subject}"
    tracking_data = load_data(f"{subject_path}/tracking.txt")

    xcorr_by_block = []

    for block, data in tracking_data.items():
        x_target = data[:,0]
        y_target = data[:,1]
        x_tracking = data[:,2]
        y_tracking = data[:,3]
        
        vel_x_target = jnp.array([jnp.diff(x_target)])
        vel_y_target = jnp.array([jnp.diff(y_target)])
        vel_x_tracking = jnp.array([jnp.diff(x_tracking)])
        vel_y_tracking = jnp.array([jnp.diff(y_tracking)])

        x_lags, x_correls = xcorr(vel_x_tracking, vel_x_target, maxlags=120)
        y_lags, y_correls = xcorr(vel_y_tracking, vel_y_target, maxlags=120)

        xcorr_by_block.append({
            "x_lags":x_lags,"x_correls":x_correls,
            "y_lags":y_lags,"y_correls":y_correls
        })
    json.dump(xcorr_by_block, open(f"{subject_path}/xcorr_by_block.json", "w"), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

for subject in range(1, 16):
    do_xcorr(1, subject)