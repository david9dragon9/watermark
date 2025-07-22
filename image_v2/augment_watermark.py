from scipy.stats import binom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import hashlib
import argparse
import numpy as np
import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, help="Path to the input image file")
    parser.add_argument("--output_path", type=str, help="Path to save the output image")
    parser.add_argument("--password", type=str, help="Password to use for watermarking")
    parser.add_argument("--mode", type=str, choices=["watermark", "detect", "both"])

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)


def calculate_p_value(N, M):
    p = 0.5
    p_value = binom.sf(N - 1, M, p)

    return p_value


class SimpleImageModel(nn.Module):
    def __init__(self, image, password: str):
        super().__init__()
        kernel_size = 10
        stride = 10

        self.password = password
        hashed_password_int = int(
            hashlib.sha256(self.password.encode("utf-8")).hexdigest(), 16
        ) % (2**32)
        set_seed(hashed_password_int)

        self.layer = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=kernel_size, stride=stride),
        )
        for param in self.layer.parameters():
            param.requires_grad = False
        self.param = torch.zeros_like(image).requires_grad_()
        new_width = int(((image.shape[1] - 2) - kernel_size) / stride + 1)
        new_height = int(((image.shape[2] - 2) - kernel_size) / stride + 1)
        self.mask = torch.randint(0, 2, (1, new_width, new_height))
        self.param_mag_weight = 20.0

    def forward(self, cat_t):
        new_cat = cat_t + self.param

        augmented_images = [
            new_cat,
            torch.flip(new_cat, [1]),
            torch.flip(new_cat, [2]),
            torch.roll(new_cat, shifts=(10, 10), dims=(1, 2)),
            new_cat + torch.randn_like(new_cat) / 255.0,
        ]
        new_cat = torch.stack(augmented_images, dim=0)

        diff = new_cat[:, :, :-2, :-2] - new_cat[:, :, 2:, 2:]
        result = self.layer(diff)

        loss = -(self.mask[None] * 2 - 1) * (result.clamp(-0.5, 0.5))
        result_sign = result > 0
        matches = (
            (result_sign.float() == self.mask[None])
            .sum(dim=(1, 2, 3))
            .min(dim=0)
            .values
        )
        return (
            loss.mean() + self.param_mag_weight * self.param.abs().mean(),
            calculate_p_value(matches.item(), result_sign[0].numel()),
        )

    def detect(self, image):
        diff = image[:, :-2, :-2] - image[:, 2:, 2:]
        result = self.layer(diff)

        result_sign = result > 0
        matches = (result_sign.float() == self.mask).sum()

        return calculate_p_value(matches.item(), result_sign.numel())


transforms_dict = {
    "orig": lambda x: x,
    "watermark": lambda x: x,
    "flipleft": lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    "flipup": lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
    "shift": lambda x: x.transform(x.size, Image.AFFINE, (1, 0, 10, 0, 1, 0)),
    "rotateright": lambda x: x.rotate(90),
    "noise1": lambda x: x.filter(ImageFilter.GaussianBlur(1)),
    "noise10": lambda x: x.filter(ImageFilter.GaussianBlur(10)),
}


def watermark_image_with_resistance(
    input_pil: Image.Image, output_pil: Image.Image, password: str, mode: str = "both"
):
    if mode in ["watermark", "both"]:
        input_image = input_pil  # Image.open(input_path)
        image_t = torch.tensor(np.array(input_image))
        image_t = image_t.permute((2, 0, 1)).float() / 255.0

        model = SimpleImageModel(image_t, password)
        optimizer = torch.optim.Adam([model.param], lr=1e-3)

        pbar = tqdm.tqdm(range(1000))
        for step in pbar:
            loss, p_value = model(image_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Step {step}, loss: {loss.item():.3f}, p-value: {p_value.item():.3f}"
            )
            if p_value < 0.01 and model.param.abs().max() < 0.03:
                break
            if p_value < 0.01:
                model.param_mag_weight += 1
            elif model.param.abs().max() < 0.01:
                model.param_mag_weight = max(1, model.param_mag_weight - 1)

        transformed_image = (
            ((image_t + model.param.detach()).permute((1, 2, 0)) * 255.0)
            .cpu()
            .detach()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        transformed_pil = Image.fromarray(transformed_image)
        if mode == "watermark":
            return transformed_pil
    if mode in ["detect", "both"]:
        loaded_pil = output_pil  # Image.open(output_path)
        loaded_t = torch.tensor(np.array(loaded_pil))
        loaded_t = loaded_t.permute((2, 0, 1)).float() / 255.0
        model = SimpleImageModel(loaded_t, password)
        return model.detect(loaded_t)


def main():
    args = get_args()
    input_pil = Image.open(args.input_path)
    if args.mode == "both":
        output_pil = watermark_image_with_resistance(
            input_pil, None, args.password, "watermark"
        )
        output_pil.save(args.output_path)
        watermark_image_with_resistance(input_pil, output_pil, args.password, "detect")
    elif args.mode == "watermark":
        output_pil = watermark_image_with_resistance(
            input_pil, None, args.password, args.mode
        )
        output_pil.save(args.output_path)
    elif args.mode == "detect":
        output_pil = Image.open(args.output_path)
        watermark_image_with_resistance(None, output_pil, args.password, args.mode)


if __name__ == "__main__":
    main()
