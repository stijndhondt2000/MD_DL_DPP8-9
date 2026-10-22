"""Backward-compatible imports for helpers moved into focused modules."""

import os
import sys

from src.utils.labels import get_binding_classes, ligand_to_binding_type, ligant_to_class
from src.utils.molecular_parsing import parse_filename, parse_pdb_from_string, remove_extension

__all__ = [
    "get_binding_classes",
    "ligand_to_binding_type",
    "ligant_to_class",
    "parse_filename",
    "parse_pdb_from_string",
    "remove_extension",
]

def calculate_num_cpus():
    """Calculate the number of CPUs to use based on environment variable or default"""
    # in python vs code debug mode, use 1 CPU to avoid issues with multiprocessing and easier debugging
    has_trace = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    has_breakpoint = sys.breakpointhook.__module__ != "sys"
    is_debug = has_trace or has_breakpoint
    if is_debug:
        print(f"Debug mode detected (has_trace={has_trace}, has_breakpoint={has_breakpoint}), using 1 CPU for easier debugging.")
        return 1
    # if running in GPULAB environment
    elif "GPULAB_CPUS_RESERVED" in os.environ:
        cpu_count = len(os.environ["GPULAB_CPUS_RESERVED"].split(","))
    # otherwise, use all available CPUs
    else: 
        cpu_count = os.cpu_count()
    cpu_count = max(0, cpu_count)
    print (f"Using {cpu_count} CPUs for data loading and processing (GPULAB_CPUS_RESERVED={os.environ.get('GPULAB_CPUS_RESERVED', 'N/A')}, total_cpus={cpu_count}).")
    return cpu_count
