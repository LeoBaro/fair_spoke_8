import os
import glob
import tarfile
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
        self.samples = []  # List of tuples: (tar_file_path, member_name)

        # Find all .tar files in the directory
        tar_files = glob.glob(os.path.join(self.tar_dir, "*.tar"), recursive=True)
        # tar_files = os.listdir(os.path.join(self.tar_dir))
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    # Iterate over members and select only image files ending with .jpg or .jpeg
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg')):
                            # Append tuple with tar file path and member name
                            self.samples.append((tar_path, member.name))
            except tarfile.TarError as e:
                print(f"Error reading {tar_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tar_path, member_name = self.samples[index]

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

        # The unique identifier here is the member name (e.g., "n04235860_14959.JPEG")
        return image, member_name, index

def custom_collate_fn(batch, image_processor):
    """
    Custom collate function to aggregate a list of tuples into batches.
    Args:
        batch (list): List of tuples (image, path, index)
    Returns:
        tuple: (data_batch, paths_batch, batch_indices)
    """
    
    # Unzip the batch into separate lists
    images, paths, indices = zip(*batch)
    inputs = image_processor(images=list(images), return_tensors="pt")
    data_batch = inputs["pixel_values"].to('cuda')

    return data_batch, list(paths), list(indices)

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
