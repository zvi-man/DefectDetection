from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from synthetic_data_creation.add_defect import add_random_defect
from utilities.display_utils import DisplayUtils
from utilities.general_utils import GeneralUtils


class DefectDataset(Dataset):
    def __init__(self, image: np.ndarray, num_images: int, transform: Optional = None):
        """
        Args:
            image (np.ndarray): Input image.
            num_images (int): Number of images to generate with defects.
        """
        self.image = image
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Generate a new image with defects
        augmented_image, defect_mask = add_random_defect(self.image)

        if self.transform is not None:
            augmented_image = self.transform(augmented_image)

        return augmented_image, defect_mask


def example():
    # Load your image, e.g., using PIL
    reference_img_path = r"../data/defective_examples/case1_reference_image.tif"
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=reference_img_path,
        to_display=True)
    # Create the dataset
    dataset = DefectDataset(reference_im, num_images=10)
    # Example of iterating through the dataset
    for i in range(len(dataset)):
        augmented_image, defect_mask = dataset[i]
        # Display the image
        DisplayUtils.display_2_images_side_by_side(augmented_image, defect_mask * 255, f"Augmented image {i}")


# Example usage
if __name__ == "__main__":
    example()
