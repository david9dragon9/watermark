from PIL import Image, ImageFilter
import argparse
import numpy as np
import json
import os
from image_v2.augment_watermark import watermark_image_with_resistance


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_folder", type=str, help="Path to the input image folder"
    )
    parser.add_argument(
        "--output_folder", type=str, help="Path to save the output image folder"
    )
    parser.add_argument(
        "--stats_output_path", type=str, help="Path to save statistics in json."
    )

    return parser.parse_args()


transforms_dict = {
    "orig": lambda x: x,
    "watermark": lambda x: x,
    "flipleft": lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    "flipup": lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
    "roll": lambda x: Image.fromarray(np.roll(np.array(x), shift=10, axis=(0, 1))),
    "noise1": lambda x: Image.fromarray(
        (np.array(x) + np.random.normal(0, 1, np.array(x).shape))
        .clip(0, 255)
        .astype(np.uint8)
    ),
}


def main():
    args = get_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    augmentations = transforms_dict.keys()
    all_p_values = {k: [] for k in augmentations}

    # Iterate through all images in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, filename)

            # Open the image
            image = Image.open(input_path)

            # Watermark the image
            watermarked_image = watermark_image_with_resistance(
                image, None, password="capstone", mode="watermark"
            )

            for augmentation in augmentations:
                if augmentation == "orig":
                    all_p_values[augmentation].append(
                        watermark_image_with_resistance(
                            None, image, password="capstone", mode="detect"
                        )
                    )
                else:
                    # Apply the augmentation
                    augmented_image = transforms_dict[augmentation](watermarked_image)

                    p_value = watermark_image_with_resistance(
                        None, augmented_image, password="capstone", mode="detect"
                    )
                    all_p_values[augmentation].append(p_value)
                    print(f"Augmentation: {augmentation}, P-value: {p_value}")

            # Save the watermarked image
            watermarked_image.save(output_path)
            print(f"Watermarked image saved to {output_path}")

    with open(args.stats_output_path, "w") as f:
        json.dump(all_p_values, f)


if __name__ == "__main__":
    main()
