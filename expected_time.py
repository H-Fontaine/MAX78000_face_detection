import re
from typing import List, Dict

OPS_OUTPUT = """
  SUMMARY OF OPS
  Hardware: 2,857,344 ops (2,741,056 macc; 116,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 867,328 ops (836,352 macc; 30,976 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 604,032 ops (557,568 macc; 46,464 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 580,800 ops (557,568 macc; 23,232 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 569,184 ops (557,568 macc; 11,616 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 234,400 ops (230,400 macc; 4,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 1,600 ops (1,600 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 16,972 bytes out of 442,368 bytes total (3.8%)
  Bias memory:   2 bytes out of 2,048 bytes total (0.1%)
"""

PROC_PER_LAYER = [3, 4, 8, 16, 32, 32]
CNN_LAYERS = [1,1,1,1,0,0]
KERNEL_SIZE = 3*3

CNN_CLOCK = 50e6 # 50 MHz

def get_opts_per_layer(ops_output: str) -> Dict[str, List[int]]:
    """Parse the output of the operations and return the number of operations per layer."""
    ops_per_layer = {
        "macc": [],
        "comp": [],
        "add": [],
        "mul": [],
        "bitwise": []
    }
    for line in ops_output.split("\n"):
        # Adjust regex to include numbers with commas
        regex_ops = re.search(r"Layer (\d+): ([\d,]+) ops \(([\d,]+) macc; ([\d,]+) comp; ([\d,]+) add; ([\d,]+) mul; ([\d,]+) bitwise\)", line)
        if regex_ops:
            ops_per_layer["macc"].append(int(regex_ops.group(3).replace(",", "")))
            ops_per_layer["comp"].append(int(regex_ops.group(4).replace(",", "")))
            ops_per_layer["add"].append(int(regex_ops.group(5).replace(",", "")))
            ops_per_layer["mul"].append(int(regex_ops.group(6).replace(",", "")))
            ops_per_layer["bitwise"].append(int(regex_ops.group(7).replace(",", "")))
    return ops_per_layer


def get_cycles_per_layer(ops_per_layer: Dict[str, List[int]], proc_per_layer: List[int], kernel_size: int, is_cnn_layer: List[int]) -> List[float]:
    """Calculate the time per layer for the entire CNN"""
    cycles_per_layer = [0] * len(ops_per_layer["macc"])
    for i in range(len(ops_per_layer["macc"])):
        macc_cycles = ops_per_layer["macc"][i] / proc_per_layer[i]
        if is_cnn_layer[i]:
            macc_cycles /= kernel_size
        cycles_per_layer[i] += macc_cycles
        cycles_per_layer[i] += ops_per_layer["comp"][i] / proc_per_layer[i]
        cycles_per_layer[i] += ops_per_layer["add"][i] / proc_per_layer[i]
        cycles_per_layer[i] += ops_per_layer["mul"][i] / proc_per_layer[i]
        cycles_per_layer[i] += ops_per_layer["bitwise"][i] / proc_per_layer[i]
    return cycles_per_layer

def get_time_per_layer(cycles_per_layer: List[float], clock: float) -> List[float]:
    """Calculate the time per layer for the entire CNN"""
    time_per_layer = [0] * len(cycles_per_layer)
    for i in range(len(cycles_per_layer)):
        time_per_layer[i] = cycles_per_layer[i] / clock
    return time_per_layer


def get_total_time(time_per_layer: List[float]) -> float:
    """Calculate the total time for the entire CNN"""
    return sum(time_per_layer)


ops_per_layer = get_opts_per_layer(OPS_OUTPUT)
print(f"Operations per layer: {ops_per_layer}")
cycles_per_layer = get_cycles_per_layer(ops_per_layer, PROC_PER_LAYER, KERNEL_SIZE, CNN_LAYERS)
print(f"Cycles per layer: {cycles_per_layer}")
time_per_layer = get_time_per_layer(cycles_per_layer, CNN_CLOCK)
print(f"Time per layer: {time_per_layer}")
total_time = get_total_time(time_per_layer)
print(f"Total time: {total_time * 1e6} us")