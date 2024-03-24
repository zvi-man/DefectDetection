import torch

from ml_method.transform import get_image_and_mask_transforms, get_image_and_mask_transforms_inference
from ml_method.unet import load_pretrained_model
from utilities.display_utils import DisplayUtils
from utilities.general_utils import GeneralUtils

if __name__ == "__main__":
    inspected_img_path = r"../data/defective_examples/case2_inspected_image.tif"
    inspected_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=inspected_img_path,
        to_display=True)
    transform = get_image_and_mask_transforms_inference()
    inspected_im_transformed, _ = transform((inspected_im, inspected_im))

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_pretrained_model()
    model = model.to(device)
    model.load_state_dict(torch.load(f"/home/zvi/Downloads/unet_19.pt"))
    model.eval()
    with torch.no_grad():
        mask = model(inspected_im_transformed.unsqueeze(0).float().to(device))
    DisplayUtils.display_2_images_side_by_side(inspected_im_transformed[0, :, :], mask[0, 0, :, :], "Inspected Image and Predicted Mask")






