import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import LogitsProcessor
import random
import copy

from scipy.stats import binom
import numpy as np
import hashlib
import argparse


def calculate_p_value(N, M):
    p = 0.5
    p_value = binom.sf(N - 1, M, p)

    return p_value


def parse_args():
    parser = argparse.ArgumentParser(description="Watermark a text with a password")

    parser.add_argument("password", type=str, help="Password to use for watermarking")
    parser.add_argument("--input_text", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    password = args.password
    seed = int(hashlib.sha256(password.encode("utf-8")).hexdigest(), 16) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PegasusForConditionalGeneration.from_pretrained(
        "tuner007/pegasus_paraphrase"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    red_list = random.sample(
        [i for i in range(model.config.vocab_size)], model.config.vocab_size // 2
    )
    red_list = torch.tensor(red_list)

    tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

    input_sentence = (
        args.input_text
    )  # "Apples are one of the most popular and versatile fruits, enjoyed around the world for their crisp texture and refreshing taste."
    generated_ids = tokenizer(
        input_sentence, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    orig_red_list = list(red_list)
    in_red_list = sum([x in orig_red_list for x in generated_ids])
    total = len(generated_ids[0])
    print("P-value", calculate_p_value(total - in_red_list, total))


if __name__ == "__main__":
    main()
