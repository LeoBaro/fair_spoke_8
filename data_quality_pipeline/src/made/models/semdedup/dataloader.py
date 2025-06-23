import os
import glob
import tarfile
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TarImageDataset(Dataset):
    def __init__(self, tar_dir, transform=None):
        """
        Args:
            tar_dir (str): Directory containing .tar files.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.tar_dir = tar_dir
        self.transform = transform
        self.samples = []  # List of tuples: (tar_file_path, member_name, uid)

        # Find all .tar files in the directory
        tar_files = glob.glob(os.path.join(self.tar_dir, "*.tar"), recursive=True)
        
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    # Extract all member names
                    member_names = [m.name for m in tar.getmembers() 
                                    if m.isfile()]
                    
                    # Find image files
                    image_files = [name for name in member_names 
                                   if name.lower().endswith(('.jpg', '.jpeg'))]
                    
                    for img_file in image_files:
                        # Find corresponding JSON file
                        json_file = os.path.splitext(img_file)[0] + ".json"
                        
                        if json_file in member_names:
                            # Extract and read the JSON file to get the UID
                            json_member = tar.getmember(json_file)
                            json_fileobj = tar.extractfile(json_member)
                            
                            if json_fileobj is not None:
                                try:
                                    metadata = json.load(json_fileobj)
                                    uid = metadata.get("uid")
                                    
                                    if uid:
                                        # Store tar path, image file name, and UID
                                        self.samples.append((tar_path, img_file, uid))
                                except json.JSONDecodeError:
                                    print(f"Error decoding JSON file {json_file} in {tar_path}")
            except tarfile.TarError as e:
                print(f"Error reading {tar_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tar_path, member_name, uid = self.samples[index]

        # Open the tar file and extract the member
        with tarfile.open(tar_path, 'r') as tar:
            try:
                member = tar.getmember(member_name)
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    raise RuntimeError(f"Failed to extract {member_name} from {tar_path}")
                image = Image.open(fileobj).convert("RGB")
            except KeyError:
                raise RuntimeError(f"Member {member_name} not found in {tar_path}")

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        # Return the image, UID and index
        return image, uid, index

def custom_collate_fn(batch, image_processor):
    """
    Custom collate function to aggregate a list of tuples into batches.
    Args:
        batch (list): List of tuples (image, path, index)
    Returns:
        tuple: (data_batch, uids_batch, batch_indices)
    """
    
    # Unzip the batch into separate lists
    images, uids, indices = zip(*batch)
    inputs = image_processor(images=list(images), return_tensors="pt")
    data_batch = inputs["pixel_values"].to('cuda')

    return data_batch, list(uids), list(indices)

class FilteredTarImageDataset(Dataset):
    def __init__(self, base_dataset, uid_list):
        """
        Filter a TarImageDataset to include only samples with UIDs in uid_list
        
        Args:
            base_dataset (TarImageDataset): The original dataset to filter
            uid_list (list): List of UIDs to include
        """
        self.base_dataset = base_dataset
        self.uid_set = set(uid_list)  # Convert to set for O(1) lookups
        
        # Create a list of indices that match our UIDs
        self.filtered_indices = []
        for i, (_, _, uid) in enumerate(base_dataset.samples):
            if uid in self.uid_set:
                self.filtered_indices.append(i)
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, index):
        # Get the sample from the base dataset using the filtered index
        base_index = self.filtered_indices[index]
        return self.base_dataset[base_index]

# if __name__ == "__main__":
#     # Example usage:
#     tar_files_directory = "/SemDeDup/data/raw"
#     batch_size = 32

#     # Optionally, you can define transforms (e.g., torchvision transforms) here.
#     transform = None

#     dataset = TarImageDataset(tar_files_directory, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

#     # Iterate over the DataLoader and get the batches.
#     for data_batch, paths_batch, batch_indices in dataloader:
#         print("Data Batch:", data_batch)          # list of image data (e.g., PIL Images or transformed tensors)
#         print("Paths Batch:", paths_batch)          # unique identifiers (filenames)
#         print("Batch Indices:", batch_indices)      # global indices for each example in the batch
#         # You can break after one batch if just testing:
#         break
