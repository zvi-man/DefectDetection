from PIL import Image
import os
from utils import load_and_display_tiff_image


def convert_tiff_to_jpg(tiff_path: str, jpg_path: str):
    try:
        with Image.open(tiff_path) as img:
            img.convert('RGB').save(jpg_path, 'JPEG')
        print("Conversion successful!")
    except Exception as e:
        print("Error during conversion:", e)


if __name__ == "__main__":
    tiff_image_path = r"data/defective_examples/case1_inspected_image.tif"
    # Provide the path to your TIFF image and the desired path for the JPG output
    inspect_im = load_and_display_tiff_image(tiff_image_path=tiff_image_path,
                                            to_display=False)
    jpg_path = tiff_image_path.replace(".tif", ".jpg")

    convert_tiff_to_jpg(tiff_image_path, jpg_path)
