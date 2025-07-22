from datasets import load_dataset
import argparse
import itertools
import json
from text_v2.vllm_watermark import vllm_watermark_generate, vllm_watermark_detect


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument(
        "--output_path", type=str, help="Path to save the generated text"
    )
    parser.add_argument(
        "--score_length",
        type=int,
        default=128256,
        help="Number of possible tokesn",
    )
    parser.add_argument("--num_ensemble", type=int, default=5)

    return parser.parse_args()


import re


def extract_last_int_in_brackets(s):
    # Find all bracketed substrings
    matches = re.findall(r"\[(.*?)\]", s)

    if not matches:
        return None  # No brackets found

    last = matches[-1]  # Get the last set of brackets

    try:
        return int(last)  # Try converting to integer
    except ValueError:
        return None  # Not an integer


def main():
    args = get_args()

    main_dataset = load_dataset("openai/gsm8k", "main")["test"][: args.num_examples]

    questions = main_dataset["question"]
    text_answers = main_dataset["answer"]

    number_answers = []
    for answer in text_answers:
        number_answers.append(int(answer.split("####")[1].strip().replace(",", "")))

    final_data = [
        {"problem": question, "answer": answer}
        for question, answer in zip(questions, number_answers)
    ]

    prompts = [
        x["problem"]
        + "\nMake sure to put your final answer in brackets. For example, if your final answer is 3, you should output [3] at the end of your answer."
        for x in final_data
    ]

    # Generate the text with watermark
    watermarked_responses = vllm_watermark_generate(
        prompts * args.num_ensemble,
        password="capstone",
        output_path=None,
        use_watermark=True,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        temperature=1.0,
    )
    print("Finished watermarked text generation.")

    vanilla_responses = vllm_watermark_generate(
        prompts,
        password="capstone",
        output_path=None,
        use_watermark=False,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
    )
    print("Finished unwatermarked text generation.")

    # Detect the watermark
    watermarked_results = vllm_watermark_detect(
        watermarked_responses,
        password="capstone",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        score_length=args.score_length,
    )
    print("Finished watermarked text detection.")

    vanilla_results = vllm_watermark_detect(
        vanilla_responses,
        password="capstone",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        score_length=args.score_length,
    )
    print("Finished unwatermarked text detection.")

    # Calculate the accuracy
    extracted_watermarked_answers = [
        extract_last_int_in_brackets(x) for x in watermarked_responses
    ]

    all_actual_answers = [[] for _ in range(len(prompts))]
    for i, extracted_answer in enumerate(extracted_watermarked_answers):
        if extracted_answer is not None:
            all_actual_answers[i % len(prompts)].append(extracted_answer)

    # Take the most common answer for each prompt
    extracted_watermarked_answers = [
        max(set(answers), key=answers.count) if answers else None
        for answers in all_actual_answers
    ]
    print("Extracted watermarked answers.")

    extracted_vanilla_answers = [
        extract_last_int_in_brackets(x) for x in vanilla_responses
    ]
    print("Extracted answers.")

    watermarked_accuracy = sum(
        [
            1 if x == y else 0
            for x, y in zip(extracted_watermarked_answers, number_answers)
        ]
    ) / len(number_answers)

    vanilla_accuracy = sum(
        [1 if x == y else 0 for x, y in zip(extracted_vanilla_answers, number_answers)]
    ) / len(number_answers)

    print(f"Watermarked Accuracy: {watermarked_accuracy}")
    print(f"Vanilla Accuracy: {vanilla_accuracy}")

    final_outputs = {
        "watermarked_accuracy": watermarked_accuracy,
        "vanilla_accuracy": vanilla_accuracy,
        "watermarked_results": watermarked_results,
        "vanilla_results": vanilla_results,
        "watermarked_responses": watermarked_responses,
        "vanilla_responses": vanilla_responses,
        "watermarked_answers": extracted_watermarked_answers,
        "vanilla_answers": extracted_vanilla_answers,
        "prompts": prompts,
    }

    with open(args.output_path, "w") as f:
        json.dump(final_outputs, f)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
