import os


from src.models.SCNN import SCNN
from src.training.training_setup import train_model
from src.config.training_config import TrainingConfig
if __name__ == "__main__":
    config_scnn = TrainingConfig(
        model_class=SCNN,
        model_args={
            "input_size": 168,
            "dropout_rate": 0.5
        },
        batch_size=4096,
        time_limit=24 * 60 * 60,  # 24 hours
        max_nr_epochs=100,
        dataset_size=1.0,  # Use the full dataset
        learning_rate = 1e-3,
    )

    train_model(config_scnn, "SCNN")

