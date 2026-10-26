from src.cli.test_model import test_model
from argparse import ArgumentParser

def scnn_test_main():
    # Parse command-line arguments
    parser = ArgumentParser(description="SCNN Test Script")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/project_antwerp/MD_DL_DPP8-9/output/models/SCNN/20261020-134127/",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="SCNN",
        help="Name of the model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (optional)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test", "external"],
        help="Data splits to evaluate (e.g., test, external)"
    )
    
    args = parser.parse_args()
    
    # Convert to the format expected by test_model if needed
    print(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Model name: {args.model_name}")
    print(f"Splits: {args.splits}")
    
    test_model(args)

if __name__ == "__main__":
    scnn_test_main()