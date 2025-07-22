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

    class WatermarkLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            orig_scores = copy.deepcopy(scores)
            softmax_scores = orig_scores.softmax(dim=1)
            if input_ids.shape[1] > 2:
                scores[:, red_list] = -10000
            greater_than = torch.where(softmax_scores > 0.9)
            scores[greater_than[0], greater_than[1]] = orig_scores[
                greater_than[0], greater_than[1]
            ]
            return scores

    tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

    input_sentence = (
        args.input_text
    )  # "Apples are one of the most popular and versatile fruits, enjoyed around the world for their crisp texture and refreshing taste."
    batch = tokenizer(input_sentence, return_tensors="pt")

    generated_ids = model.generate(
        batch["input_ids"],
        logits_processor=[WatermarkLogitsProcessor()],
        max_new_tokens=512,
    )
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_sentence)

    orig_red_list = list(red_list)
    in_red_list = sum([x in orig_red_list for x in generated_ids[0][1:-1]])
    total = len(generated_ids[0]) - 2
    print("P-value", calculate_p_value(total - in_red_list, total))


if __name__ == "__main__":
    main()
