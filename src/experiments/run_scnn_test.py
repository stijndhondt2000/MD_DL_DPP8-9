"""
SCNN Test runner that accepts checkpoint path programmatically.

Usage:
    python run_scnn_test.py --checkpoint /path/to/checkpoint.pt --splits test external
    python run_scnn_test.py /path/to/checkpoint.pt test external  # positional args too
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run SCNN model testing programmatically")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth/.pt)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Data splits to evaluate (e.g., test, external)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (optional)"
    )
    
    args = parser.parse_args()
    
    # Build command for the underlying test_model script
    cmd = [
        sys.executable,
        "/home/sam/Documents/Projects/MD_DL_DPP8-9/src/experiments/test_scnn.py",
        "--checkpoint", args.checkpoint,
    ]
    
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    cmd.extend(["--splits"] + args.splits)
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
