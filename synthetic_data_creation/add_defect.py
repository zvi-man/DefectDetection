from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from utilities.general_utils import GeneralUtils

# Constants
NOTABLE_PIXEL_DIFF = 20


def add_defect(image, defect_size_x: int = 5, defect_size_y: int = 5,
               defect_intensity: int = 100, is_defect_plus: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add a synthetic defect to the input image and generate a corresponding mask.

    Parameters:
        image (numpy.ndarray): 2D array representing the SEM image.
        defect_size_x (int): Size of the defect kernel (default is 5).
        defect_size_y (int): Size of the defect kernel (default is 5).
        defect_intensity (float): Intensity of the defect (default is 0.5).
        is_defect_plus (bool): True if the defect is to be added to the image (default is True).

    Returns:
        tuple: A tuple containing the image with the added defect and the corresponding mask.
    """
    # Create a random position for the defect
    height, width = image.shape
    x = np.random.randint(defect_size_x, height - defect_size_x)
    y = np.random.randint(defect_size_y, width - defect_size_y)

    # Create defect kernel
    # TODO:
    #  1. add option for covariances matrix that is not diagonal
    #  2. add option for non gaussian like defects
    mu = np.array([defect_size_x / 2, defect_size_y / 2])  # Mean
    covariance_matrix = np.array([[int(defect_size_x / 3), 0],
                                  [0, int(defect_size_y / 3)]])  # Covariance matrix
    x_def, y_def = np.mgrid[range(defect_size_x), range(defect_size_y)]
    pos = np.dstack((x_def, y_def))

    # Create the 2D Gaussian distribution
    gaussian = multivariate_normal(mu, covariance_matrix)

    # Calculate the probability density function (PDF) values for each point on the grid
    gaussian_pdf = gaussian.pdf(pos)
    defect_kernel = np.uint8(defect_intensity * gaussian_pdf / gaussian_pdf.max())

    # Add the defect to the image
    image_with_defect = image.copy()
    x_offset = x - int(defect_size_x / 2)
    y_offset = y - int(defect_size_y / 2)
    original_patch = image_with_defect[x_offset:x_offset + defect_size_x, y_offset:y_offset + defect_size_y].copy()
    if is_defect_plus:
        patch_with_defect = np.clip(original_patch.astype(int) + defect_kernel.astype(int), 0, 255).astype(np.uint8)
    else:
        patch_with_defect = np.clip(original_patch.astype(int) - defect_kernel.astype(int), 0, 255).astype(np.uint8)
    image_with_defect[x_offset:x_offset + defect_size_x, y_offset:y_offset + defect_size_y] = patch_with_defect

    # Create a mask for the defect
    defect_mask = np.zeros_like(image_with_defect)
    defect_kernel_mask = np.abs(patch_with_defect.astype(int) - original_patch.astype(int)) > NOTABLE_PIXEL_DIFF
    defect_mask[x_offset:x_offset + defect_size_x, y_offset:y_offset + defect_size_y] = defect_kernel_mask

    return image_with_defect, defect_mask * 255


def add_random_defect(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add a random defect to the input image and generate a corresponding mask.

    Parameters:
        image (numpy.ndarray): 2D array representing the SEM image.

    Returns:
        tuple: A tuple containing the image with the added defect and the corresponding mask.
    """
    height, width = image.shape
    defect_size_x = np.random.randint(3, int(height / 2))
    defect_size_y = np.random.randint(3, int(width / 2))
    defect_intensity = np.random.randint(1, 255)
    is_defect_plus = np.random.choice([True, False])
    try:
        return add_defect(image, defect_size_x=defect_size_x, defect_size_y=defect_size_y,
                          defect_intensity=defect_intensity, is_defect_plus=is_defect_plus)
    except:
        print(f"Failed to add a random defect with size {defect_size_x}, {defect_size_y}, "
              f"{defect_intensity}, {is_defect_plus}")
        raise


def example_add_defect():
    # Example usage:
    # Generate a synthetic SEM image
    reference_img_path = r"../data/defective_examples/case1_reference_image.tif"
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=reference_img_path,
        to_display=True)

    # Add a defect to the SEM image and get the defect mask
    sem_with_defect, defect_mask = add_random_defect(reference_im)

    # Display the original SEM image, SEM image with defect, and the defect mask
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(reference_im, cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[1].imshow(sem_with_defect, cmap='gray')
    axes[1].set_title('SEM Image with Defect')
    axes[2].imshow(defect_mask, cmap='gray')
    axes[2].set_title('Defect Mask')
    plt.show()


if __name__ == "__main__":
    example_add_defect()
