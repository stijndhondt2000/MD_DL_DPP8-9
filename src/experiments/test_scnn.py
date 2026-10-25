


from src.cli.test_model import test_model


def scnn_test_main():
    # create an args object with the necessary attributes for testing
    class Args:
        def __init__(self):
            self.checkpoint_path = "/project_antwerp/MD_DL_DPP8-9/output/models/SCNN/20261020-134127/"
            self.model_name = "SCNN"
            self.output_dir = None
            self.splits = ["test", "external"]
    args = Args()
    test_model(
        args
    )
if __name__ == "__main__":
    scnn_test_main()