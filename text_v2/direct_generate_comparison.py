import argparse
from text_v2.vllm_watermark import vllm_watermark_generate, vllm_watermark_detect
import os
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question_input_path", type=str, help="Path to the input question file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save the responses, statistics, and figures",
    )
    parser.add_argument(
        "--password1", type=str, help="Password 1 to use for watermarking"
    )
    parser.add_argument(
        "--password2", type=str, help="Password 2 to use for watermarking"
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

    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    nonwatermarked_responses = vllm_watermark_generate(
        question_input_path=args.question_input_path,
        output_dir=args.output_dir,
        password=args.password,
        model_name=args.model_name,
        hash_token_window=args.hash_token_window,
        use_watermark=False,
    )

    watermarked_responses1 = vllm_watermark_generate(
        question_input_path=args.question_input_path,
        output_dir=args.output_dir,
        password=args.password1,
        model_name=args.model_name,
        hash_token_window=args.hash_token_window,
        use_watermark=True,
    )

    watermarked_responses2 = vllm_watermark_generate(
        question_input_path=args.question_input_path,
        output_dir=args.output_dir,
        password=args.password2,
        model_name=args.model_name,
        hash_token_window=args.hash_token_window,
        use_watermark=True,
    )

    response_data = []
    for x, y, z in zip(
        nonwatermarked_responses, watermarked_responses1, watermarked_responses2
    ):
        response_data.append(
            {"nonwatermarked": x, "watermarked1": y, "watermarked2": z}
        )

    response_data_path = os.path.join(args.output_dir, "response_data.json")
    with open(response_data_path, "w") as f:
        json.dump(response_data, f)

    nonwatermarked_p_values = vllm_watermark_detect(
        responses=nonwatermarked_responses,
        password=args.password,
        hash_token_window=args.hash_token_window,
    )

    watermarked_p_values1 = vllm_watermark_detect(
        responses=watermarked_responses1,
        password=args.password1,
        hash_token_window=args.hash_token_window,
    )

    watermarked_p_values2 = vllm_watermark_detect(
        responses=watermarked_responses2,
        password=args.password2,
        hash_token_window=args.hash_token_window,
    )


if __name__ == "__main__":
    main()
