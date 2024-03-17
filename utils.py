import cv2
import numpy as np


def load_and_display_tiff_image(tiff_image_path: str,
                                grayscale: bool = True,
                                to_display: bool = True) -> np.ndarray:
    """Loads a tiff image and displays it.

    Args:
        tiff_image_path (str): The path to the tiff image.

    Returns:
        np.ndarray: The loaded image.
    """
    image = cv2.imread(tiff_image_path)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if to_display:
        display_image(image, tiff_image_path)
    return image


def display_2_images_side_by_side(image1: np.ndarray, image2: np.ndarray) -> None:
    """Displays two images side-by-side.

    Args:
        image1 (np.ndarray): The first image.
        image2 (np.ndarray): The second image.
    """
    display_image(np.hstack((image1, image2)), "Images side-by-side")


def display_image(image: np.ndarray, title: str) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def median_and_gaussian_blur(image: np.ndarray, median_kernel_size: int = 3,
                             gaussian_kernel_size: int = (5, 5), gaussian_sigmaX: int = 0):
    # Apply median filter
    median_blurred = cv2.medianBlur(image, median_kernel_size)

    # Apply Gaussian blur
    gaussian_blurred = cv2.GaussianBlur(median_blurred, gaussian_kernel_size, gaussian_sigmaX)

    return gaussian_blurred


def register_images(gray_im1, gray_im2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and extract descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray_im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_im2, None)

    # img = cv2.drawKeypoints(gray_im1, keypoints1, gray_im1,
    #                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # display image with keypoints
    # display_image(img, "Image with keypoints")

    # Match descriptors between the two images
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate affine transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Apply transformation to image2
    registered_image = cv2.warpAffine(gray_im2, transformation_matrix, (gray_im1.shape[1], gray_im1.shape[0]))

    return registered_image


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


if __name__ == "__main__":
    inspect_im = load_and_display_tiff_image(tiff_image_path=r"data/defective_examples/case1_inspected_image.tif",
                                             to_display=False)
    reference_im = load_and_display_tiff_image(tiff_image_path=r"data/defective_examples/case1_reference_image.tif",
                                               to_display=False)
    # display_2_images_side_by_side(inspect_im, reference_im)
    # register the 2 images using sift
    # registered_reference_im = register_images(median_and_gaussian_blur(inspect_im),
    #                                           median_and_gaussian_blur(reference_im))
    registered_reference_im = shift_image(reference_im, -24, 4)
    display_2_images_side_by_side(inspect_im, registered_reference_im)
    diff = np.abs(median_and_gaussian_blur(inspect_im) - median_and_gaussian_blur(registered_reference_im))
    display_image(diff, title="Difference")
    a = 1
