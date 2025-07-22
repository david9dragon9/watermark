from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import LogitsProcessor, AutoTokenizer
from pydantic import BaseModel
import copy
import json
import torch
import random
import hashlib
import numpy as np
import hashlib
import argparse
from scipy.stats import binom
import math


def set_seed(seed: int):
    torch.manual_seed(seed)


def calculate_p_value(N, M):
    p = 0.5
    p_value = binom.sf(N - 1, M, p)

    return p_value


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--password", type=str, help="Password to use for watermarking")
    parser.add_argument(
        "--input_prompt_path", type=str, help="Path to the input prompt file"
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the generated text"
    )
    parser.add_argument(
        "--use_watermark", action="store_true", help="Whether to use watermarking"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-1B-Instruct",
        help="Model name to use for generating text",
    )
    parser.add_argument(
        "--hash_token_window",
        type=int,
        default=4,
        help="Number of tokens to hash for watermarking",
    )
    parser.add_argument(
        "--score_length",
        type=int,
        default=128256,
        help="Number of possible tokesn",
    )

    return parser.parse_args()


class OnTheFlyLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        vocab_size: int,
        hashed_password_int: int = 0,
        hash_token_window: Optional[int] = 4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hashed_password_int = hashed_password_int
        self.hash_token_window = hash_token_window
        self.weights = []
        for i in range(hash_token_window):
            self.weights.insert(0, self.vocab_size**i)
        self.weights = torch.tensor(self.weights)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        hashed_password_int = self.hashed_password_int
        if len(input_ids) >= self.hash_token_window and len(scores) > 0:
            softmax_scores = scores.softmax(dim=-1)

            last_input_ids = input_ids[-self.hash_token_window :]
            hashed_input_ids = (
                sum(x * y for x, y in zip(self.weights, last_input_ids))
                + hashed_password_int
            )
            set_seed(hashed_input_ids % (2**32))
            red_list = (torch.randint(0, 2, [len(scores)]) > 0).to(
                softmax_scores.device
            )

            scores[red_list & (softmax_scores < 0.8)] = -10000
        return scores

    def detect_watermark(
        self,
        text,
        tokenizer_name,
        score_length: int = 128256,
        override_hashed_password_int: Optional[int] = None,
    ):
        hashed_password_int = (
            override_hashed_password_int
            if override_hashed_password_int is not None
            else self.hashed_password_int
        )
        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        input_ids = self.tokenizer(text)["input_ids"]
        in_red = 0
        for i in range(4, len(input_ids) - 1):
            curr_input_ids = input_ids[:i]
            last_input_ids = curr_input_ids[-self.hash_token_window :]
            hashed_input_ids = (
                sum(x * y for x, y in zip(self.weights, last_input_ids))
                + hashed_password_int
            )
            set_seed(hashed_input_ids % (2**32))
            red_list = torch.randint(0, 2, [score_length]) > 0

            if red_list[input_ids[i]]:
                in_red += 1
        total = len(input_ids) - 4
        return 1.0 if len(input_ids) <= 4 else calculate_p_value(total - in_red, total)


def templatize(prompts, tokenizer):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True
        )
        for x in prompts
    ]


class VLLMWatermarkGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model_name, dtype="float16")

    def watermark(self, prompt: str, password: str, hash_token_window: int = 4):
        hashed_password_int = int(
            hashlib.sha256(password.encode("utf-8")).hexdigest(), 16
        ) % (2**32)

        logits_processors = [
            OnTheFlyLogitsProcessor(
                vocab_size=self.tokenizer.vocab_size,
                hashed_password_int=hashed_password_int,
                hash_token_window=hash_token_window,
            )
        ]

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=512, logits_processors=logits_processors
        )

        chat_prompt = templatize([prompt], self.tokenizer)[0]
        results = self.llm.generate(chat_prompt, sampling_params=sampling_params)
        return results[0].outputs[0].text

    def detect(self, response: str, password: str, hash_token_window: int = 4):
        hashed_password_int = int(
            hashlib.sha256(password.encode("utf-8")).hexdigest(), 16
        ) % (2**32)

        logits_processor = OnTheFlyLogitsProcessor(
            vocab_size=self.tokenizer.vocab_size,
            hashed_password_int=hashed_password_int,
            hash_token_window=hash_token_window,
        )

        return logits_processor.detect_watermark(response, self.model_name)


def vllm_watermark_generate(
    prompts: List[str],
    password: str,
    output_path: Optional[str] = None,
    use_watermark: bool = True,
    model_name: str = "meta-llama/Llama-3.1-1B-Instruct",
    hash_token_window: Optional[int] = 4,
    temperature: float = 0.0,
    json_schema: Optional[BaseModel] = None,
    batch_size: Optional[int] = None,
    choices: Optional[List] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat_prompts = templatize(prompts, AutoTokenizer.from_pretrained(model_name))

    llm = LLM(model_name, dtype="float16")

    hashed_password_int = int(
        hashlib.sha256(password.encode("utf-8")).hexdigest(), 16
    ) % (2**32)

    if use_watermark:
        logits_processors = [
            OnTheFlyLogitsProcessor(
                vocab_size=tokenizer.vocab_size,
                hashed_password_int=hashed_password_int,
                hash_token_window=hash_token_window,
            )
        ]
    else:
        logits_processors = None

    guided_decoding_params = GuidedDecodingParams(
        json=json_schema,
        choice=choices,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        logits_processors=logits_processors,
        guided_decoding=guided_decoding_params if json_schema is not None else None,
    )
    if batch_size is not None:
        text_results = []
        for i in range(math.ceil(len(chat_prompts) / batch_size)):
            text_results.extend(
                [
                    x.outputs[0].text
                    for x in llm.generate(
                        chat_prompts[batch_size * i : batch_size * (i + 1)],
                        sampling_params=sampling_params,
                    )
                ]
            )
            import gc

            gc.collect()
    else:
        results = llm.generate(chat_prompts, sampling_params=sampling_params)
        text_results = [x.outputs[0].text for x in results]

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(text_results, f)

    return text_results


def vllm_watermark_detect(
    responses: List[str],
    password: str,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    hash_token_window: Optional[int] = 4,
    score_length: int = 128256,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hashed_password_int = int(
        hashlib.sha256(password.encode("utf-8")).hexdigest(), 16
    ) % (2**32)

    logits_processor = OnTheFlyLogitsProcessor(
        vocab_size=tokenizer.vocab_size,
        hashed_password_int=hashed_password_int,
        hash_token_window=hash_token_window,
    )

    return [logits_processor.detect_watermark(x, model_name) for x in responses]


def main():
    args = get_args()

    with open(args.input_prompt_path, "r") as f:
        prompts = json.load(f)

    responses = vllm_watermark_generate(
        prompts=prompts,
        password=args.password,
        output_path=args.output_path,
        use_watermark=args.use_watermark,
        model_name=args.model_name,
        hash_token_window=args.hash_token_window,
    )

    scores = vllm_watermark_detect(
        responses=responses,
        password=args.password,
        model_name=args.model_name,
        hash_token_window=args.hash_token_window,
        score_length=args.score_length,
    )

    print(scores)


if __name__ == "__main__":
    main()
