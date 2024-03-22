import cv2
import numpy as np
from utilities.display_utils import DisplayUtils


class GeneralUtils:
    @staticmethod
    def load_and_display_tiff_image(tiff_image_path: str,
                                    grayscale: bool = True,
                                    to_display: bool = True) -> np.ndarray:
        """Loads a tiff image and displays it.

        Args:
            tiff_image_path (str): The path to the tiff image.

        Returns:
            np.ndarray: The loaded image.
            :param grayscale:
            :param tiff_image_path:
            :param to_display:
        """
        image = cv2.imread(tiff_image_path)
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if to_display:
            DisplayUtils.display_image(image, tiff_image_path)
        return image

    @staticmethod
    def subtract_2_images_only_non_zero_pixels(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Subtracts 2 images.

        Args:
            img1 (np.ndarray): The first image.
            img2 (np.ndarray): The second image.
        """
        mask_im1 = img1 == 0
        mask_im2 = img2 == 0
        diff_im = np.clip(np.abs(img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)
        diff_im[mask_im1] = 0
        diff_im[mask_im2] = 0
        return diff_im
