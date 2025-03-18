import os
import subprocess

def main():
    # List of "thread" counts to test
    thread_counts = [1, 2,3,4,5,6,7,8,9,10,11,12]

    for nc in thread_counts:
        # Create a copy of the current environment
        env = os.environ.copy()
        
        # Control OMP-based parallelism:
        # - OMP_NUM_THREADS: maximum OpenMP threads
        # - MKL_NUM_THREADS: if NumPy is using MKL, set that too
        #env["OMP_NUM_THREADS"] = str(nc)
        #env["OMP_DYNAMIC"] = "FALSE"   # disable dynamic adjustment
        #env["MKL_NUM_THREADS"] = str(nc)
        #env["MKL_DYNAMIC"] = "FALSE"

        # We can also set XLA_FLAGS to disable multi-threaded eigen so that
        # we rely purely on OMP for threading. Some JAX versions respond to this:
        #env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

        rep = 1
        task_difficulty = 10000
        code = fr'''
import time
import jax
import jax.numpy as jnp
import numpy as np
import numpyro

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count({nc})
jax_core_num = jax.local_device_count()

# Define a function for matrix multiplication
def matmul(A, B):
    return jnp.dot(A, B)

size = {task_difficulty}  # Adjust the matrix size if needed
A = jax.random.normal(jax.random.PRNGKey(0), (size, size))
B = jax.random.normal(jax.random.PRNGKey(1), (size, size))

# Warm-up JIT
_ = matmul(A, B).block_until_ready()

times = []
rep = {rep}  # Reduce repetition since matrix multiplication is slower
for _ in range(rep):
    t0 = time.time()
    _ = matmul(A, B).block_until_ready()
    times.append(time.time() - t0)

avg_time = np.mean(times)
print("JAX cores="+str(jax_core_num)+", average runtime over {rep} runs: "+str(avg_time)+" s")
'''

        # Run the snippet in a fresh subprocess
        subprocess.run(["python", "-c", code], env=env, check=True)

if __name__ == "__main__":
    main()
