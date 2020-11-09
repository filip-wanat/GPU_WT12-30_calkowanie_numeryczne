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

step = 0.000001
# make array size from start to end step by step
step_array = np.arange(0, 30.0, step)
# make sinus values
sin_array = np.sin(step_array)
# empty array for results, same size as input
out = np.empty_like(sin_array)

threads_per_block = 512  # determine threads per block
blocks_per_grid = 256  # determine blocks per grid

timer = time.perf_counter()  # start timer
# calculate partials numerial in CPU
for position, item in enumerate(sin_array):
    out[position] = item * step
timer = time.perf_counter() - timer  # end timer
sum = np.sum(out)  # sum all values to obtain numerial from sinus

print(f"Sum: {sum}")
print(f"Normal Time: {timer}")

timer = time.perf_counter()  # start timer
numerical[blocks_per_grid, threads_per_block](sin_array, step, out)  # start kernel on GPU with static defined values
timer = time.perf_counter() - timer  # end timer
sum = np.sum(out)  # sum all values to obtain numerial from sinus

print(f"Sum: {sum}")
print(f"CUDA Time: {timer}")
