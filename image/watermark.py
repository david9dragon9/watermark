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

    parser.add_argument("image", type=str, help="Path to the image to watermark")
    parser.add_argument("password", type=str, help="Password to use for watermarking")
    parser.add_argument("output", type=str, help="Path to save the watermarked image")

    parser.add_argument(
        "--optimize-steps", type=int, default=1000, help="Number of optimization steps"
    )
    parser.add_argument(
        "--difference-loss-weight",
        type=float,
        default=0.1,
        help="Weight of the difference loss",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image = Image.open(args.image)
    original_size = image.size
    image.resize((224, 224))
    image = np.array(image)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.Sequential(*list(model.children())[:-1])

    optimize_steps = args.optimize_steps
    difference_loss_weight = args.difference_loss_weight
    learning_rate = args.learning_rate

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
    # random_weights = torch.softmax(random_weights, dim=0).cuda()

    image_param = torch.tensor(image, dtype=torch.float32)
    difference_param = torch.nn.parameter.Parameter(
        torch.zeros_like(image_param, requires_grad=True)
    )
    optimizer = torch.optim.Adam([difference_param], lr=learning_rate)

    preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()

    pbar = tqdm.tqdm(range(optimize_steps))
    for i in pbar:
        optimizer.zero_grad()

        watermarked_image = (image_param + difference_param) / 255.0
        watermarked_image = watermarked_image.permute(2, 0, 1)
        watermarked_image = torch.clamp(watermarked_image, 0, 1)

        watermarked_image = preprocess(watermarked_image)
        watermarked_image = watermarked_image.unsqueeze(0).to(device)
        embeddings = model(watermarked_image)
        embeddings = torch.sigmoid(embeddings)
        # probabilities = F.softmax(logits, dim=1)

        score_loss = -torch.sum(embeddings.flatten() * random_weights.flatten())
        difference_loss = difference_loss_weight * torch.mean(difference_param**2)
        total_loss = score_loss + difference_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pbar.set_description(
            f"Step {i+1}/{optimize_steps}, Total: {total_loss.item():.2f}, Score: {-score_loss.item():.2f}, Difference L2: {torch.mean(difference_param ** 2).item():.2f}"
        )

    watermarked_image = image_param + difference_param
    watermarked_image = torch.clamp(watermarked_image, 0, 255)

    watermarked_image = watermarked_image.detach().numpy().astype(np.uint8)
    watermarked_image = Image.fromarray(watermarked_image)
    watermarked_image.resize(original_size)
    watermarked_image.save(args.output)


if __name__ == "__main__":
    main()
