import os
import tarfile

import numpy as np
from Bio import PDB
from pathlib import Path


def extract_coordinates(pdb_file, pdb_id):
    """Extract 3D coordinates from PDB file object."""
    from io import StringIO

    # Convert bytes to string
    content = pdb_file.read().decode('utf-8')

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(content))

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord)

    return np.array(coords, dtype=np.float32)


def construct_pdb_record(pdb_id, dpp_class, ligand_name, binding_type, coords, replica_id):
    """Construct a Hugging Face-style dictionary for a single PDB record."""
    return {
        'pdb_id': pdb_id,
        'dpp_class': dpp_class,
        'ligand_name': ligand_name,
        'binding_type': binding_type,
        'coordinates': coords,
        'num_atoms': len(coords),
        'replica_id': replica_id
    }


def parse_pdb_streaming(tar=None, frames=None, dpp_class=None, ligand_name=None, binding_type=None, replica_id=None, tar_path=None):
    """Yield one dataset record per PDB file stored inside a TAR archive.

    The generator accepts either an already-open ``tarfile.TarFile`` object or a
    path to a TAR archive. Passing the archive path is important when running
    ``Dataset.from_generator`` with multiprocessing, because the live tar handle
    is not reliably serializable across worker processes.
    """
    close_tar = False
    if tar is None:
        if tar_path is None:
            raise ValueError("Either tar or tar_path must be provided.")
        tar = tarfile.open(tar_path, mode='r:*')
        close_tar = True
    elif isinstance(tar, (str, os.PathLike)):
        tar_path = str(tar)
        tar = tarfile.open(tar_path, mode='r:*')
        close_tar = True

    try:
        tar_members = frames if frames is not None else tar.getmembers()
        for frame in tar_members:
            member_name = getattr(frame, 'name', '')
            if not member_name.endswith('.pdb'):
                continue

            extracted = tar.extractfile(frame)
            if extracted is None:
                continue

            try:
                with extracted:
                    pdb_id = Path(member_name).stem
                    coords = extract_coordinates(extracted, pdb_id)
                    yield construct_pdb_record(pdb_id, dpp_class, ligand_name, binding_type, coords, replica_id)
            except Exception as exc:
                print(f"Error parsing {pdb_id}: {exc}")
                continue
    finally:
        if close_tar:
            tar.close()


def extract_pdb_files_from_directory(pdb_dir: str, dpp_class: str|None = None, ligand_name: str|None = None, binding_type: str|None = None, replica_id: int|None = None) -> list:
    """
    Extract PDB files from a directory and generate preprocessed data samples.
    :param pdb_dir: Path to the directory containing PDB files.
    :param dpp_class: DPP class. Optional
    :param ligand_name: Name of the ligand. Optional
    :param binding_type: Type of binding. Optional
    :param replica_id: ID of the replica. Optional
    :return: List of preprocessed data samples ready for interpretability analysis.
    Each sample is a dictionary containing {'coordinates': list, 'binding_type': str, 'dpp_class': str, 'ligand_name': str, 'replica_id': int, num_atoms: int}
    """
    pdb_files = Path(pdb_dir).glob('*.pdb')
    samples = []
    for pdb_file in pdb_files:
        sample = extract_pdb_file(str(pdb_file), ligand_name=ligand_name, binding_type=binding_type, dpp_class=dpp_class, replica_id=replica_id)
        samples.append(sample)
    return samples


def extract_pdb_file(pdb_file: str, ligand_name: str|None = None, binding_type: str|None = None, dpp_class: str|None = None, replica_id: int|None = None) -> dict:
    """
    Generate the preprocessed data sample from a PDB file.
    Preprocess the PDB file and extract relevant features
    :param pdb_file: Path to the PDB file to be processed.
    :param ligand_name: Name of the ligand. Optional
    :param binding_type: Type of binding. Optional
    :param dpp_class: DPP class. Optional
    :param replica_id: ID of the replica. Optional
    :return: Preprocessed data sample ready for interpretability analysis. 
    This is a sample containing {'coordinates': list, 'binding_type': str, 'dpp_class': str, 'ligand_name': str, 'replica_id': int, num_atoms: int}
    """
    with open(pdb_file, 'rb') as f:
        pdb_id = Path(pdb_file).stem
        coords = extract_coordinates(f, pdb_id)
    return construct_pdb_record(pdb_id, dpp_class, ligand_name, binding_type, coords, replica_id)