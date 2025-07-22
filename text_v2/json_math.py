from datasets import load_dataset
from pydantic import BaseModel
import argparse
import json
from text_v2.vllm_watermark import vllm_watermark_generate, vllm_watermark_detect
import math


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

    return parser.parse_args()


import re


def extract_last_int_in_brackets(s):
    # Find all bracketed substrings
    matches = re.findall(r"\[(.*?)\]", s)

    if not matches:
        return ""  # No brackets found

    last = matches[-1]  # Get the last set of brackets

    try:
        return int(last)  # Try converting to integer
    except ValueError:
        return ""  # Not an integer


class AnswerSchema(BaseModel):
    reasoning: str
    answer: int


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
        + "\nMake sure to output your answer in a json format with keys reasoning and answer. You should first reason about the problem before outputting a final answer."
        for x in final_data
    ]

    # Generate the text with watermark
    watermarked_responses = []
    for i in range(math.ceil(len(prompts) / 20)):
        watermarked_responses.extend(
            vllm_watermark_generate(
                prompts[20 * i : 20 * (i + 1)],
                password="capstone",
                output_path=None,
                use_watermark=True,
                model_name="meta-llama/Llama-3.2-1B-Instruct",
                hash_token_window=4,
                json_schema=AnswerSchema.model_json_schema(),
            )
        )
    print("Finished watermarked text generation.")

    vanilla_responses = vllm_watermark_generate(
        prompts,
        password="capstone",
        output_path=None,
        use_watermark=False,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        json_schema=AnswerSchema.model_json_schema(),
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
    extracted_watermarked_answers = []
    for response in watermarked_responses:
        try:
            json_answer = json.loads(response)["answer"]
            extracted_watermarked_answers.append(json_answer)
        except:
            extracted_watermarked_answers.append(None)

    extracted_vanilla_answers = []
    for response in vanilla_responses:
        try:
            json_answer = json.loads(response)["answer"]
            extracted_vanilla_answers.append(json_answer)
        except:
            extracted_vanilla_answers.append(None)
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
