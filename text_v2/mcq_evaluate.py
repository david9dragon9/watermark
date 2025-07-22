from datasets import load_dataset
from text_v2.vllm_watermark import vllm_watermark_generate, vllm_watermark_detect
import argparse
import json
import os
import re


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated questions",
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to use from the dataset",
    )

    return parser.parse_args()


def extract_choice_from_response(response):
    # Extract the last choice from the response
    matches = re.findall(r"\[(.*?)\]", response)

    if not matches:
        return None  # No brackets found

    last_choice = matches[-1]  # Get the last set of brackets

    return last_choice.strip()  # Return the choice without brackets


def main():
    args = get_args()

    dataset = load_dataset("satoshidg/GSM-MC-Stage")["test"][: args.num_examples]

    # Dataset has Question row and rows A, B, C, D. There is also an Answer row which is either A, B, C, or D.
    # Put them together into a single prompt with all four choices and then log which one is correct.
    prompts = []
    for i in range(len(dataset["Question"])):
        question = dataset["Question"][i]
        answer = dataset["Answer"][i]
        choices = [dataset["A"][i], dataset["B"][i], dataset["C"][i], dataset["D"][i]]
        prompt = f"{question}\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}"
        prompts.append({"prompt": prompt, "answer": answer})

    final_prompts = [
        f"{prompt}\nMake sure to put your final answer in brackets. For example, if your final answer is A, you should output [A] at the end of your answer."
        for prompt in prompts
    ]

    watermarked_responses = vllm_watermark_generate(
        final_prompts,
        output_path=args.output_path,
        password="mcq",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        use_watermark=True,
    )

    vanilla_responses = vllm_watermark_generate(
        final_prompts,
        output_path=args.output_path,
        password="mcq",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        use_watermark=False,
    )

    # Evaluate. First extract the choice from the watermarked responses
    watermarked_choices = [
        extract_choice_from_response(response) for response in watermarked_responses
    ]
    vanilla_choices = [
        extract_choice_from_response(response) for response in vanilla_responses
    ]

    # Compare the choices to the actual answers
    watermarked_correct = [
        1 if choice == prompt["answer"] else 0
        for choice, prompt in zip(watermarked_choices, prompts)
    ]
    vanilla_correct = [
        1 if choice == prompt["answer"] else 0
        for choice, prompt in zip(vanilla_choices, prompts)
    ]

    # Calculate accuracy
    watermarked_accuracy = sum(watermarked_correct) / len(watermarked_correct)
    vanilla_accuracy = sum(vanilla_correct) / len(vanilla_correct)

    print(f"Watermarked accuracy: {watermarked_accuracy * 100:.2f}%")
    print(f"Vanilla accuracy: {vanilla_accuracy * 100:.2f}%")

    # Save the results
    all_results = {
        "watermarked_responses": watermarked_responses,
        "vanilla_responses": vanilla_responses,
        "watermarked_choices": watermarked_choices,
        "vanilla_choices": vanilla_choices,
        "watermarked_accuracy": watermarked_accuracy,
        "vanilla_accuracy": vanilla_accuracy,
    }

    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
