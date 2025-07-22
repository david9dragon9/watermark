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
        f"{prompt}\nSolve this problem. Make sure to output reasoning before your final answer."
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

    watermarked_prompts = [
        f"{prompt}\nHere is the reasoning and answer: {response}. Now, simply output a choice of answer A, B, C, or D and nothing else."
        for prompt, response in zip(prompts, watermarked_responses)
    ]

    watermarked_choices = vllm_watermark_generate(
        watermarked_prompts,
        output_path=args.output_path,
        password="mcq",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        use_watermark=False,
        choices=["A", "B", "C", "D"],
    )

    # Compare the choices to the actual answers
    watermarked_correct = [
        1 if choice == prompt["answer"] else 0
        for choice, prompt in zip(watermarked_choices, prompts)
    ]

    # Calculate accuracy
    watermarked_accuracy = sum(watermarked_correct) / len(watermarked_correct)

    print(f"Watermarked accuracy: {watermarked_accuracy * 100:.2f}%")

    # Save the results
    all_results = {
        "watermarked_responses": watermarked_responses,
        "watermarked_choices": watermarked_choices,
        "watermarked_accuracy": watermarked_accuracy,
    }

    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
