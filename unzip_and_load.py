import os
import re
import zipfile
import torch
import warnings
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CircuitDataset(Dataset):
    """PyTorch Dataset for circuit graph data with fixed-size tensors.
    
    Each sample contains:
        - Node features (X)
        - Edge features (E)
        - Circuit performance (Y)
        - Node masks (node_mask)
    """
    def __init__(self, data_list):
        """
        Args:
            data_list: List[dict] where each dictionary contains:
                - X: [22, 21] node features (float32)
                - E: [22, 22, 25] edge features (float32)
                - Y: [13] circuit performance (float32)
                - node_mask: [22] node masks (bool)
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return { 
            'X': torch.from_numpy(sample['X']).float(),  # convert to torch.tensor
            'E': torch.from_numpy(sample['E']).float(),
            'Y': torch.from_numpy(sample['Y']).float(),
            'node_mask': torch.from_numpy(sample['node_mask']).bool()  
        }


def unzip_with_progress(zip_path, target_dir):
    """Extract ZIP file with progress tracking and clean up source file after completion.
    
    Args:
        zip_path: Path to source ZIP file
        target_dir: Destination directory for extracted files
        
    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.infolist()
            total_size = sum(file.file_size for file in file_list)
            
            with tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024,
                desc=f"Extracting {os.path.basename(zip_path)}"
            ) as pbar:
                for file in file_list:
                    zip_ref.extract(file, target_dir)
                    pbar.update(file.file_size)
        
        tqdm.write(f"Removed source file: {os.path.basename(zip_path)}")
        return True
        
    except Exception as e:
        tqdm.write(f"Failed to extract {os.path.basename(zip_path)}: {str(e)}")
        return False


def unzip_all_data_chunks():
    """Chunk processing for all data chunk ZIP files in current directory."""
    print("Starting data chunk extraction of data_chunks_*.zip files...")
    pattern = re.compile(r'^data_chunks_\d+\.zip$')
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, "unzipped_data")
    
    os.makedirs(target_dir, exist_ok=True)
    tqdm.write(f"Extraction directory: {target_dir}")

    zip_files = [f for f in os.listdir(current_dir) if pattern.match(f)]
    if not zip_files:
        tqdm.write("No data_chunks_*.zip files found")
        return

    with tqdm(total=len(zip_files), unit="file", desc="Overall progress") as main_pbar:
        success_count = 0
        for zip_file in zip_files:
            zip_path = os.path.join(current_dir, zip_file)
            if unzip_with_progress(zip_path, target_dir):
                success_count += 1
            main_pbar.update(1)
    
    tqdm.write(f"\nExtraction complete: {success_count}/{len(zip_files)} files processed\n")

def load_data(dir_path='unzipped_data'):
    """Load and concatenate all circuit data from extracted files.
    
    Args:
        dir_path: The directory path of unzipped data, default to 'unzipped_data'
    Returns:
        CircuitDataset: Initialized dataset object
    """
    # Suppress FutureWarnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    files = [f for f in os.listdir(dir_path)]
    files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    data_list = []
    with tqdm(files, desc="Loading chunks", unit="file") as pbar:
        for file in pbar:
            file_path = os.path.join(dir_path, file)
            # Display current filename
            pbar.set_postfix(file=file)
            # Load data (with simulated sub-progress)
            data_chunk = torch.load(file_path)
            data_list.extend(data_chunk)

    circuit_dataset = CircuitDataset(data_list) 
    print("Circuit dataset is created and returned")
    return circuit_dataset
            
            
if __name__ == "__main__":
    unzip_all_data_chunks() # extract data chunk (execute only once)

