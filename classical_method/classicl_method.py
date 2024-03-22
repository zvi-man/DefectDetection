import numpy as np
import cv2
import matplotlib.pyplot as plt

from utilities.augmentation_utils import AugmentationUtils
from utilities.display_utils import DisplayUtils
from utilities.general_utils import GeneralUtils


def binarize_register_images(gray_im1, gray_im2):
    binary_image1 = AugmentationUtils.binarize_image(gray_im1)
    binary_image2 = AugmentationUtils.binarize_image(gray_im2)

    DisplayUtils.display_2_images_side_by_side(binary_image1, binary_image2)

    # Find contours in the binary images
    contours1, _ = cv2.findContours(binary_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the bounding boxes of the contours
    x1, y1, w1, h1 = cv2.boundingRect(contours1[0])
    x2, y2, w2, h2 = cv2.boundingRect(contours2[0])

    # Calculate the translation to align the bounding boxes
    dx = x1 - x2
    dy = y1 - y2

    # Apply translation to the second binary image
    rows, cols = binary_image2.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    registered_binary_image = cv2.warpAffine(gray_im2, m, (cols, rows))

    return registered_binary_image


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


def experiments():
    inspect_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=r"../data/defective_examples/case2_inspected_image.tif",
        to_display=False)
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=r"../data/defective_examples/case2_reference_image.tif",
        to_display=False)
    # display_2_images_side_by_side(inspect_im, reference_im)
    # register the 2 images using sift
    # registered_reference_im = register_images(median_and_gaussian_blur(inspect_im),
    #                                           median_and_gaussian_blur(reference_im))

    registered_reference_im = AugmentationUtils.shift_image(reference_im, -6, -5)
    # registered_reference_im = AugmentationUtils.shift_image(reference_im, -24, 4)

    # registered_reference_im = binarize_register_images(median_and_gaussian_blur(inspect_im),
    #                                                    median_and_gaussian_blur(reference_im))
    DisplayUtils.display_2_images_side_by_side(AugmentationUtils.median_and_gaussian_blur(inspect_im),
                                               AugmentationUtils.median_and_gaussian_blur(registered_reference_im))
    diff = GeneralUtils.subtract_2_images_only_non_zero_pixels(AugmentationUtils.median_and_gaussian_blur(inspect_im),
                                                               AugmentationUtils.median_and_gaussian_blur(
                                                                   registered_reference_im))
    DisplayUtils.display_image(diff, title="Difference")

    threshold_value = 40  # Adjust threshold value as needed
    ret, diff_binary_img = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    DisplayUtils.display_image(diff_binary_img, title="Difference binary")

    kernel = np.ones((3, 3), np.uint8)
    diff_binary_img = cv2.erode(diff_binary_img, kernel, iterations=1)
    diff_binary_img = cv2.dilate(diff_binary_img, kernel, iterations=1)

    DisplayUtils.display_image(diff_binary_img, title="Difference binary")


def edge_experiments():
    inspect_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=r"../data/defective_examples/case1_inspected_image.tif",
        to_display=True)
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=r"../data/defective_examples/case1_reference_image.tif",
        to_display=True)

    # display image histograms
    hist1, bins1 = np.histogram(inspect_im.ravel(), bins=256, range=[0, 256])
    hist2, bins2 = np.histogram(reference_im.ravel(), bins=256, range=[0, 256])

    # Plot histograms
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.title('Grayscale Histogram - Image 1')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist1, color='black')
    plt.xlim([0, 256])
    plt.xticks(np.arange(0, 256, step=50))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.subplot(2, 1, 2)
    plt.title('Grayscale Histogram - Image 2')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist2, color='black')
    plt.xlim([0, 256])
    plt.xticks(np.arange(0, 256, step=50))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # Displaying values on x-axis
    plt.xticks(np.arange(0, 256, step=50))  # Adjust this according to your preference
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # run edge detection
    blurred_image = cv2.GaussianBlur(inspect_im, (3, 3), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred_image, 50, 150)  # You can adjust these threshold values

    DisplayUtils.display_image(edges, title="Edges")


if __name__ == "__main__":
    experiments()
