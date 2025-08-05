# Dependencies for the Dataloader module 
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image
import matplotlib.pyplot as plt



class Unet_Dataset(Dataset):
    """
    A custom PyTorch Dataset class for loading paired images used in U-Net models.
    
    This class assumes you have two directories: one for the first set of images (e.g., inputs),
    and one for the second set of images (e.g., targets or ground truths).
    
    Args:
        frist_dir (str): Path to the first image directory.
        second_dir (str): Path to the second image directory.
        index_pairs (list of tuple, optional): List of index pairs (i, j) indicating which images 
                                               to load from each directory. Defaults to aligned pairs.
        Height (int): The height to which images will be resized.
        Width (int): The width to which images will be resized.
    """
    
    def __init__(self, frist_dir, second_dir, index_pairs=None, Height=224, Width=224):
        self.frist_dir_path = frist_dir
        self.second_dir_path = second_dir
        self.height = Height
        self.width = Width

        # Load and sort image paths from both directories
        self.frist_images = sorted([
            os.path.join(self.frist_dir_path, f) 
            for f in os.listdir(self.frist_dir_path)
        ])
        self.second_images = sorted([
            os.path.join(self.second_dir_path, f) 
            for f in os.listdir(self.second_dir_path)
        ])

        # Define preprocessing transformations (Resize and convert to tensor)
        self.transform = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor()
        ])

        # Generate default index pairs (i, i) if not provided
        if index_pairs is None:
            self.index_pairs = [
                (i, i) for i in range(min(len(self.frist_images), len(self.second_images)))
            ]
        else:
            self.index_pairs = index_pairs

    def __getitem__(self, idx):
        """
        Fetches a pair of transformed images using the provided index.
        
        Args:
            idx (int): Index of the image pair to load.
        
        Returns:
            tuple: (frist_image_tensor, second_image_tensor)
        """
        frist_index, second_index = self.index_pairs[idx]

        # Load and convert images to RGB format
        frist_image = Image.open(self.frist_images[frist_index]).convert("RGB")
        second_image = Image.open(self.second_images[second_index]).convert("RGB")

        # Apply transformations to images
        frist_image_tensor = self.transform(frist_image)
        second_image_tensor = self.transform(second_image)

        return frist_image_tensor, second_image_tensor

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.
        """
        return len(self.index_pairs)





def load_batch_from_dataset(dataset: Dataset):
    """
    Loads the full batch of frist and second tensors based on index_pairs
    Returns: frist_batch, second_batch of shape (B, C, H, W)
    """
    frist_list, second_list = [], []
    for i in range(len(dataset)):
        frist_tensor, second_tensor = dataset[i]
        frist_list.append(frist_tensor)
        second_list.append(second_tensor)
    
    frist_batch = torch.stack(frist_list)  # (B, C, H, W)
    second_batch = torch.stack(second_list)    # (B, C, H, W)
    return frist_batch, second_batch


def fuse_batched_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Concatenates two batched tensors along the channel dimension.
    
    Args:
        tensor1 (Tensor): A tensor of shape (B, C, H, W)
        tensor2 (Tensor): A tensor of shape (B, C, H, W)
        
    Returns:
        Tensor: A fused tensor of shape (B, 2C, H, W)
    """
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape"
    return torch.cat((tensor1, tensor2), dim=1)  # Concatenate along channel dimension



def move_tensor_to_device(tensor: torch.Tensor, device=None) -> torch.Tensor:
    """
    Moves a single tensor to the specified device (GPU or CPU).
    
    Args:
        tensor (Tensor): A PyTorch tensor.
        device (torch.device, optional): The target device. If None, auto-selects CUDA if available.

    Returns:
        Tensor: The tensor on the target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return tensor.to(device)


