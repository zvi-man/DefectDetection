import numpy as np
from tqdm import tqdm

from classical_method.classicl_method import classical_inspect_images_numpy
from evaluation.metrics import f1_score_masks
from ml_method.transform import RandomShiftWithMask
from synthetic_data_creation.synthetic_dataset import DefectDataset
from utilities.display_utils import DisplayUtils
from utilities.general_utils import GeneralUtils

if __name__ == "__main__":
    reference_img_path = r"../data/defective_examples/case1_reference_image.tif"
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=reference_img_path,
        to_display=True)
    # Create the dataset
    num_images = 10
    dataset = DefectDataset(reference_im, num_images=num_images, transform=RandomShiftWithMask(0.1))
    f1_scores = []
    for inspected_image, gt_mask in tqdm(dataset):
        predicted_mask = classical_inspect_images_numpy(inspected_image, reference_im)
        DisplayUtils.display_2_images_side_by_side(predicted_mask, gt_mask)
        f1_scores.append(f1_score_masks(gt_mask.flatten(), predicted_mask.flatten(), pos_label=255))
        print(f"For classic method, F1 score: {np.mean(f1_scores)}")


