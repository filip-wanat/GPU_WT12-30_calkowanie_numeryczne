from numba import cuda


@cuda.jit
def numerical(x, step, out):
    tx = cuda.threadIdx.x  # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x  # number of blocks in the grid

    start = tx + ty * block_size
    stride = block_size * grid_size

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] * step


import numpy as np
import time

# must be 
step = 0.000001

step_array = np.arange(0, 6.30, step)  # around 2 * pi
sin_array = np.sin(step_array)
out = np.empty_like(sin_array)

threads_per_block = 256
blocks_per_grid = 128

timer = time.perf_counter()
for position, item in enumerate(sin_array):
    out[position] = item * step
sum = np.sum(out)
timer = time.perf_counter() - timer

print(f"Sum: {sum}")
print(f"Normal Time: {timer}")

timer = time.perf_counter()
numerical[blocks_per_grid, threads_per_block](sin_array, step, out)
sum = np.sum(out)
timer = time.perf_counter() - timer

print(f"Sum: {sum}")
print(f"CUDA Time: {timer}")
