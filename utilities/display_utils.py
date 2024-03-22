import numpy as np
import cv2


class DisplayUtils:
    @classmethod
    def display_2_images_side_by_side(cls, image1: np.ndarray, image2: np.ndarray) -> None:
        """Displays two images side-by-side.

        Args:
            image1 (np.ndarray): The first image.
            image2 (np.ndarray): The second image.
        """
        cls.display_image(np.hstack((image1, image2)), "Images side-by-side")

    @staticmethod
    def display_image(image: np.ndarray, title: str) -> None:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
