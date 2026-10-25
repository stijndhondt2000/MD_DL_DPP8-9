from pathlib import Path

from dotenv import load_dotenv
from src.interpretability.interpretability_executer import execute_interpretability
from src.utils.utils import ligant_to_class
import time
import os

def count_significant_residues(input_folder: Path = Path("/home/stijn/Sam_ModelVisualisation/output_interpretability_final/output2"),
                            output_folder_root: Path = Path("/home/stijn/Sam_ModelVisualisation/output_interpretability_final/output4"),   
                            base_pdb_file_folder: Path = Path("/home/stijn/Sam_ModelVisualisation/output_interpretability_final/"),
                            methods: list[str] = ["saliency", "occlusion", "integrated_gradients"]):
                                                                                                                                                                                                                                                                            
    models = None
    for dpp_dir in input_folder.iterdir():
        dpp = dpp_dir.stem
        # ligand
        for ligand_dir in dpp_dir.iterdir():
            # loop over the interpretability methods
            for replica_dir in ligand_dir.iterdir():
                residue_dir = {}
                ligand = ligand_dir.name
                models_dirs = replica_dir.iterdir() # get the names of the models
                model_names = [model_dir.name for model_dir in models_dirs if model_dir.is_dir()]
                for model in model_names:
                    residue_dir[model] = {}
                    for method in methods:
                        residue_dir[model][method] = {}
                    imagefolder = Path(replica_dir, model, "images")

                    for pdb_file in imagefolder.rglob("*.pdb"):
                        if "average" in str(pdb_file).lower():
                            continue
                        method = None
                        for poss_method in methods:
                            if poss_method in str(pdb_file).lower():
                                method = poss_method

                        
                        with open(pdb_file, "r") as fin:
                            for line in fin:
                                if line.startswith(("ATOM", "HETATM")):
                                        residue_name = line[17:27].strip().upper()
                                        b_factor = line[60:67].strip().upper()

                                        if b_factor == "100.00":
                                            if not residue_name in residue_dir[model][method].keys():
                                                residue_dir[model][method][residue_name] = 0
                                            residue_dir[model][method][residue_name] += 1
            for model in model_names:
                for method, residue_name_dict in residue_dir[model].items():
                    max_counts = max(residue_name_dict.values())
                    for  residue_name in residue_name_dict.keys():
                        residue_dir[model][method][residue_name] /= max_counts
                    pdb_base_file = base_pdb_file_folder / dpp_dir.name / ligand_dir.name / "replica1"
                    pdb_base_file = next(pdb_base_file.rglob("*frame_1000.pdb"))
                    output_folder = output_folder_root / dpp / ligand / model
                    os.makedirs(output_folder, exist_ok=True)
                    name = f"{dpp}_{ligand}_{model}_{method}"
                    pdb_name = f"{name}.pdb"
                    pml_name = f"{name}.pml"
                    txt_name = f"{name}.txt"
                    with open(pdb_base_file, "r") as fin, open(output_folder / pdb_name, "w") as fout, open(output_folder / pml_name, "w") as pmlout:
                        for line in fin:
                            if line.startswith(("ATOM", "HETATM")):
                                    residue_name = line[17:27].strip().upper()
                                    if residue_name in residue_dir[model][method].keys():
                                        residue_b_factor = residue_dir[model][method][residue_name] *100
                                    else:
                                        residue_b_factor = 0
                                    line = line[:60] + f"{residue_b_factor:6.2f}" + line[66:]


                            fout.write(line)
                    
                        pml_string = (
                            f"load {name}.pdb, {name}\n"
                            f"color gray, {name}\n"
                            f"spectrum b, gray70 yellow orange red, {name}\n"
                            f"show cartoon, {name}\n"
                            f"select {name}_high_atoms, ({name} and polymer.protein and b > 50.00)\n"
                            f"select {name}_high_residue, byres {name}_high_atoms\n"
                            f"hide cartoon, {name}_high_residue\n"
                            f"show sticks, {name}_high_residue\n"
                        )
                        pmlout.write(pml_string)
                    with open(output_folder / txt_name, "w") as txtout:
                        lines = []
                        for residue_name in residue_dir[model][method].keys():
                            residue_b_factor = residue_dir[model][method][residue_name] *100
                            if residue_b_factor>50.0:
                                lines.append((residue_name[4], residue_name[0:4], residue_name[6:9]))
                        lines.sort(key=lambda x: (x[0], int(x[2]), x[1]))
                        for line in lines:
                            txtout.write(f"{line[0]} {line[1]} {line[2]}\n")

                



if __name__ == "__main__":    
    load_dotenv()
    root_input_dir = Path("/project_antwerp/dataset/decompressed_both_datasets_every1000/")
    root_output_dir = Path(os.getenv("OUTPUT_DIR", default="/project_antwerp/MD_DL_DPP8-9/output/"))
    interpretability_output_dir = root_output_dir / "interpretability"
    all_frames_output_dir = interpretability_output_dir / "all_frames"
    aggregated_replicas_output_dir = interpretability_output_dir / "aggregated_replicas"
    count_significant_residues(
        input_folder=all_frames_output_dir,
        output_folder_root=aggregated_replicas_output_dir,
        base_pdb_file_folder=root_input_dir,
        methods=list(["saliency"])
    )