import numpy as np
import cv2


class AugmentationUtils:
    @staticmethod
    def binarize_image(gray_im):
        # Apply adaptive thresholding
        _, binary_image = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary_image

    @staticmethod
    def median_and_gaussian_blur(image: np.ndarray, median_kernel_size: int = 3,
                                 gaussian_kernel_size: int = (5, 5), gaussian_sigmaX: int = 0):
        # Apply median filter
        median_blurred = cv2.medianBlur(image, median_kernel_size)

        # Apply Gaussian blur
        gaussian_blurred = cv2.GaussianBlur(median_blurred, gaussian_kernel_size, gaussian_sigmaX)

        return gaussian_blurred

    @staticmethod
    def shift_image(image: np.ndarray, w: int, h: int) -> np.ndarray:
        shifted_image = np.zeros_like(image)

        # Calculate the shifted indices
        rows, cols = image.shape[:2]
        start_row, start_col = max(0, h), max(0, w)
        end_row, end_col = min(rows + h, rows), min(cols + w, cols)

        # Copy the original image to the shifted indices
        shifted_image[start_row:end_row, start_col:end_col] = image[max(0, -h):min(rows, rows - h),
                                                              max(0, -w):min(cols, cols - w)]

        return shifted_image
