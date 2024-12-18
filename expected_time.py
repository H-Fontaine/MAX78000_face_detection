import argparse
import re
import os
from typing import List, Dict
import yaml

def count_processors(processor_mask: str) -> int:
    """Count the number of processors used from a processor mask."""
    # Remove the "0x" prefix and split the mask into groups
    binary_representation = bin(int(processor_mask.replace(".", ""), 16))[2:]
    return binary_representation.count("1")

def parse_yaml(file_path: str):
    """Parse the YAML file to extract processor usage and convolutional layers."""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    
    proc_per_layer = []
    cnn_layers = []
    
    for layer in data.get("layers", []):
        # Check if the layer is convolutional
        is_conv = layer.get("op") == "conv2d"
        cnn_layers.append(1 if is_conv else 0)
        
        # Count the number of processors used
        processor_mask = layer.get("processors", "0x0")
        proc_per_layer.append(count_processors(processor_mask))
    
    return proc_per_layer, cnn_layers



def get_opts_per_layer(file_path: str) -> Dict[str, List[int]]:
    """Parse the output of the operations and return the number of operations per layer."""
    with open(file_path, "r") as file:
        data = file.read()

    ops_per_layer = {
        "macc": [],
        "comp": [],
        "add": [],
        "mul": [],
        "bitwise": []
    }
    for line in data.split("\n"):
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

KERNEL_SIZE = 3*3
CNN_CLOCK = 50e6 # 50 MHz

if __name__ == "__main__":
    # Get arguments from the command line
    parser = argparse.ArgumentParser(description="Calculate the expected time for a CNN model")
    parser.add_argument("model", type=str, help="The model to analyze")
    args = parser.parse_args()

    proc_per_layer, cnn_layers = parse_yaml(f"models/{args.model}/config.yaml")
    ops_per_layer = get_opts_per_layer(f"models/{args.model}/ops.txt")
    cycles_per_layer = get_cycles_per_layer(ops_per_layer, proc_per_layer, KERNEL_SIZE, cnn_layers)
    time_per_layer = get_time_per_layer(cycles_per_layer, CNN_CLOCK)
    total_time = get_total_time(time_per_layer)

    # write the everything to file
    with open(f"models/{args.model}/expected_inference_time.txt", "w") as file:
        file.write(f"Total time: {total_time * 10**6:.2f} us\n")
        file.write("\n")
        file.write("Time per layer:\n")
        for i, time in enumerate(time_per_layer):
            file.write(f"Layer {i + 1}: {time * 10**6:.2f} us\n")
        file.write("\n")
        file.write("Cycles per layer:\n")
        for i, cycles in enumerate(cycles_per_layer):
            file.write(f"Layer {i + 1}: {cycles:.2f} cycles\n")
        file.write("\n")
        file.write("Operations per layer:\n")
        for key in ops_per_layer:
            file.write(f"{key.capitalize()}: {', '.join([f'{op:_}' for op in ops_per_layer[key]])}\n")
        file.write("\n")
        file.write("Processors per layer:\n")
        for i, proc in enumerate(proc_per_layer):
            file.write(f"Layer {i + 1}: {proc} processors\n")
        file.write("\n")
        file.write("CNN layers:\n")
        for i, is_cnn in enumerate(cnn_layers):
            file.write(f"Layer {i + 1}: {'Yes' if is_cnn else 'No'}\n")