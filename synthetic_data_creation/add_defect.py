import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from utilities.general_utils import GeneralUtils


def add_defect(image, defect_size_x: int = 5, defect_size_y: int = 5,
               defect_intensity: int = 100):
    """
    Add a synthetic defect to the input image and generate a corresponding mask.

    Parameters:
        image (numpy.ndarray): 2D array representing the SEM image.
        defect_size_x (int): Size of the defect kernel (default is 5).
        defect_size_y (int): Size of the defect kernel (default is 5).
        defect_intensity (float): Intensity of the defect (default is 0.5).

    Returns:
        tuple: A tuple containing the image with the added defect and the corresponding mask.
    """
    # Create a random position for the defect
    height, width = image.shape
    x = np.random.randint(defect_size_x, height - defect_size_x)
    y = np.random.randint(defect_size_y, width - defect_size_y)

    # Create a mountain-like defect kernel
    defect_kernel = np.zeros((defect_size_x, defect_size_y), dtype=np.uint8)
    mu = np.array([defect_size_x / 2, defect_size_y / 2])  # Mean
    covariance_matrix = np.array([[int(defect_size_x/3), 0],
                                  [0, int(defect_size_x/3)]])  # Covariance matrix
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
    patch_with_defect = np.clipZ?Dyy[[[[[[[[[[[[[[[[[[[[efect_mask


def example_add_defect():
    # Example usage:
    # Generate a synthetic SEM image
    reference_img_path = r"../data/defective_examples/case1_reference_image.tif"
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=reference_img_path,
        to_display=True)

    # Add a defect to the SEM image and get the defect mask
    sem_with_defect, defect_mask = add_defect(reference_im, defect_size_x=30, defect_size_y=30,
                                              defect_intensity=100)

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
