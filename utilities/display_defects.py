import cv2

from utilities.general_utils import GeneralUtils
from utilities.display_utils import DisplayUtils


def plot_defects(image_path: str, defects: list):
    inspect_im = GeneralUtils.load_and_display_tiff_image(tiff_image_path=image_path,
                                                          to_display=True)
    for x, y in defects:
        cv2.circle(inspect_im, (x, y), 5, (0, 0, 255), -1)
    DisplayUtils.display_image(inspect_im, "Inspected image")


if __name__ == "__main__":
    # Image 1
    defects = [[149, 334], [82, 245], [97, 82]]
    image_path = r"../data/defective_examples/case1_inspected_image.tif"
    plot_defects(image_path, defects)

    defects = [[344, 265], [105, 108], [80, 262]]
    image_path = r"../data/defective_examples/case2_inspected_image.tif"
    plot_defects(image_path, defects)
