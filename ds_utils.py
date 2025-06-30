import os
from Bio.PDB import PDBParser
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np


# * Preprocessing to get CA coordinates
# Directories
PDB_DIR = Path("./dataset/cath-S40-pdb")
SAVE_DIR = Path("./dataset/ca_coords")

# Function to extract CA coordinates
def extract_ca_coordinates(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].coord)
    return np.array(coords)


# Get CA coordinates dataset
def get_ca_coordinates():
    print("Getting CA coordinates")
    for pdb_file in PDB_DIR.iterdir():
        coords = extract_ca_coordinates(pdb_file)
        if len(coords) >= 10:  # Skip very short domains
            save_path = SAVE_DIR / (pdb_file.stem + ".npy")
            np.save(save_path, coords)
    print("Done")


# * Pytorch dataset and dataloaders
class CADataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = list(Path(data_dir).glob("*.npy"))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        coords = np.load(self.data_files[idx])
        coords = coords - coords.mean(axis=0)
        scale = np.linalg.norm(coords, axis=1).mean()
        coords = coords / scale
        return torch.tensor(coords, dtype=torch.float32)


def get_dataloaders(batch_size=32):
    if os.path.exists(SAVE_DIR):
        print(f"Directory already exists: {SAVE_DIR} — skipping coordinate extraction.")
    else:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        get_ca_coordinates()

    # Create dataset
    dataset = CADataset("dataset/ca_coords")

    # Split: 80% train, 10% val, 10% test
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Collate for variable-length (just return a list)
    def collate_fn(batch):
        return batch  # List[Tensor], each of shapes [L_i, 3]

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader