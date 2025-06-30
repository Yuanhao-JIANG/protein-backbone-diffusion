from ftplib import FTP
import os
import urllib.request
import tarfile
import subprocess


# * List repository contents
def list_ftp_dir(host, path):
    ftp = FTP(host)
    ftp.login()  # anonymous login

    ftp.cwd(path)
    files = []
    ftp.retrlines('LIST', files.append)

    print(f"\nContents of ftp://{host}/{path}/:\n")
    for line in files:
        parts = line.split()
        if len(parts) >= 9:
            size = int(parts[4]) / (1024 * 1024)  # bytes to MB
            name = parts[-1]
            print(f"{name:<50} {size:.2f} MB")

    ftp.quit()

# list_ftp_dir("orengoftp.biochem.ucl.ac.uk", "cath/releases/latest-release/non-redundant-data-sets")
# list_ftp_dir("orengoftp.biochem.ucl.ac.uk", "cath/releases/latest-release/cath-classification-data")

# * Download datasets
# Target dataset folder
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Base FTP URL for CATH
BASE_URL = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/"

# Files to download: (relative_path, save_as)
files_to_download = [
    # 3D structures
    ("non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz", "cath-S40.pdb.tgz"),
    
    # 1D Amino Acid Sequences
    ("non-redundant-data-sets/cath-dataset-nonredundant-S40.fa", "cath-S40.fa"),
    
    # 1D Atom sequences
    ("non-redundant-data-sets/cath-dataset-nonredundant-S40.atom.fa", "cath-S40.atom.fa"),
    
    # Domain list
    ("non-redundant-data-sets/cath-dataset-nonredundant-S40.list", "cath-S40.list"),

    # Domain metadata for claasification
    ("cath-classification-data/cath-domain-description-file.txt", "cath-domain-description.txt"),
    ("cath-classification-data/cath-domain-list.txt", "cath-domain-list.txt"),
]

# Download function
def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"Data already exists: {save_path}")
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")

# Loop through files
for rel_path, filename in files_to_download:
    url = BASE_URL + rel_path
    local_path = os.path.join(DATASET_DIR, filename)
    try:
        download_file(url, local_path)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

# * Extract .tgz archive
archive_path = "dataset/cath-S40.pdb.tgz"
extract_path = "dataset/cath-S40-pdb"

# Extract .tgz archive
if os.path.exists(extract_path):
    print(f"Directory already exists: {extract_path} — skipping extraction.")
else:
    print(f"Extracting {archive_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    subprocess.run("mv ./* ../ && cd .. && rm -rf dompdb", shell=True, cwd=os.path.join(extract_path, "dompdb"))
    print(f"Extraction complete.")