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


def register_images(gray_im1, gray_im2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and extract descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray_im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_im2, None)

    # Match descriptors between the two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Apply transformation to image2
    registered_image = cv2.warpAffine(gray_im2, transformation_matrix, (gray_im1.shape[1], gray_im1.shape[0]))

    return registered_image


if __name__ == "__main__":
    inspect_im = load_and_display_tiff_image(tiff_image_path=r"data/defective_examples/case1_inspected_image.tif",
                                             to_display=False)
    reference_im = load_and_display_tiff_image(tiff_image_path=r"data/defective_examples/case1_reference_image.tif",
                                               to_display=False)
    display_2_images_side_by_side(inspect_im, reference_im)
    # register the 2 images using sift
    registered_reference_im = register_images(inspect_im, reference_im)
    display_2_images_side_by_side(inspect_im, registered_reference_im)
    diff = np.abs(inspect_im - registered_reference_im)
    display_image(diff, title="Difference")
