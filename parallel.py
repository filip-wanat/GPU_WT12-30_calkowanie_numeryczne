from math import floor
import time
import numpy as np
from numba import cuda
from utils.print_line import Printer

STEP = 0.000001
END = 30.0

threads_per_block = 512  # determine threads per block
blocks_per_grid = 256  # determine blocks per grid


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


def cpu(fun_array, step, out):
    for position, item in enumerate(fun_array):
        out[position] = item * step


def gpu_cpu_split(fun_array, step, out, gpu_usage, END):
    # find index that will split fun_array into two
    gpu_last_index = floor(gpu_usage*END/step)
    # split fun_array accroding to index
    gpu_array = fun_array[:gpu_last_index]
    cpu_array = fun_array[gpu_last_index:]

    # make operations on cpu
    cpu_out = np.empty_like(cpu_array)
    cpu(cpu_array, step, cpu_out)

    # make operations on gpu
    gpu_out = np.empty_like(gpu_array)
    numerical[blocks_per_grid, threads_per_block](gpu_array, step, gpu_out)

    # combine arrays
    output = np.empty_like(fun_array)
    output[:gpu_last_index] = gpu_out
    output[gpu_last_index:] = cpu_out

    return output


def main(END):
    with Printer(f"result/result-{int(END)}", "csv") as printer:
        print(f"END:\t\t\t{END}")
        print(f"STEP:\t\t\t{STEP}\n")
        printer.print(f"Sum;Percent GPU Time [s];time;STEP:{STEP};END:{END}\n")

        # make array size from start to end step by step
        step_array = np.arange(0, END, STEP)
        # make sinus values
        sin_array = np.sin(step_array)
        # empty array for results, same size as input
        out = np.empty_like(sin_array)

        timer = time.perf_counter()  # start timer
        # calculate partials numerial in CPU
        out = np.empty_like(out)
        cpu(sin_array, STEP, out)
        timer = time.perf_counter() - timer  # end timer
        sum = np.sum(out)  # sum all values to obtain numerial from sinus
        print(f"Sum:\t\t\t{sum}")
        print(f"0.0% GPU Time [s]:\t{timer}\n")
        printer.printValues(sum,0,timer)

        for SPLIT in np.arange(0.05, 1.0, 0.05):

            out = np.empty_like(sin_array)
            timer = time.perf_counter()
            output = gpu_cpu_split(sin_array, STEP, out, SPLIT, END)
            timer = time.perf_counter() - timer
            sum = np.sum(output)

            print(f"Sum:\t\t\t{sum}")
            print(f"{floor(SPLIT*100)}.0% GPU Time [s]:\t{timer}\n")
            printer.printValues(sum,floor(SPLIT*100),timer)

        timer = time.perf_counter()  # start timer
        # start kernel on GPU with static defined values
        out = np.empty_like(sin_array)
        numerical[blocks_per_grid, threads_per_block](sin_array, STEP, out)
        timer = time.perf_counter() - timer  # end timer
        timer2 = time.perf_counter()
        sum = np.sum(out)  # sum all values to obtain numerial from sinus
        timer2 = time.perf_counter()-timer2
        print(f"Sum:\t\t\t{sum}")
        print(f"100.0% GPU Time [s]:\t{timer}\n")
        printer.printValues(sum,100,timer)


for END in np.arange(1.0, 21.0, 1.0):
    main(END)
