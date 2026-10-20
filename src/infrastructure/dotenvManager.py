from fileinput import filename
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path
def construct_cache_identifier(used_percentage, prefix="mapped"):
    return f"{prefix}_{used_percentage}"

class dotenvManager:
    
    def __init__(self, dotenv_path=".env"):
        self.dotenv_path = dotenv_path
        load_dotenv(dotenv_path)
        # Check if env was loaded successfully
        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(f"{dotenv_path} not found. Please ensure the .env file exists.")


    def get_env_variable(self, var_name, default=None, required=False):
        value = os.getenv(var_name, default)
        if required and value is None:
            raise ValueError(f"Environment variable {var_name} is required but not set.")
        return value

    def set_env_variable(self, var_name, value):
        os.environ[var_name] = value
        with open(self.dotenv_path, "a") as f:
            f.write(f"{var_name}={value}\n")
    def get_safetensors_directory(self):
        return self.get_env_variable("SAFETENSORS_DIR", required=True)

    def get_dataset_directory(self):
        return self.get_env_variable("DATASET_DIR", required=True)
    def get_compressed_dataset_directory(self):
        return self.get_env_variable("COMPRESSED_DATASET_DIR", required=True)