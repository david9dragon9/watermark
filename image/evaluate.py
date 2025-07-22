from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm
import hashlib
from torchvision.models import resnet50, ResNet50_Weights
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Watermark an image with a password")

    parser.add_argument("password", type=str, help="Password to use for watermarking")
    parser.add_argument("output", type=str, help="Path to save the watermarked image")

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    image = Image.open(args.output)
    image.resize((224, 224))
    image = np.array(image)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.Sequential(*list(model.children())[:-1])

    password = args.password
    seed = int(hashlib.sha256(password.encode("utf-8")).hexdigest(), 16) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_weights = torch.randn(
        [
            2048,
        ],
        dtype=torch.float32,
        requires_grad=False,
    ).cuda()

    image_param = torch.tensor(image, dtype=torch.float32)
    preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()

    watermarked_image = image_param
    watermarked_image = watermarked_image.permute(2, 0, 1) / 255.0
    watermarked_image = torch.clamp(watermarked_image, 0, 1)

    watermarked_image = preprocess(watermarked_image)
    watermarked_image = watermarked_image.unsqueeze(0).to(device)

    embeddings = model(watermarked_image)
    # probabilities = F.softmax(logits, dim=1)

    score_loss = -torch.sum(embeddings.flatten() * random_weights.flatten())
    print(f"Score loss: {score_loss.item():.3f}")

    aligned_count = (embeddings.flatten() > 0) * (random_weights.flatten() > 0)
    aligned_prop = aligned_count.mean()
    print(f"Aligned proportion: {aligned_prop.item():.3f}")


if __name__ == "__main__":
    main()
