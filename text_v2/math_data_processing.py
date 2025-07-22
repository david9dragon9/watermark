from datasets import load_dataset
import json
import argparse
import random


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated questions",
    )
    parser.add_argument(
        "--num_subset_examples",
        type=int,
        default=1000,
        help="Number of examples to use from the dataset",
    )
    parser.add_argument(
        "--subset_output_path",
        type=str,
        default="math_data.json",
        help="Path to save the subset of the dataset",
    )

    return parser.parse_args()


def main():
    args = get_args()
    dataset = load_dataset("openai/gsm8k")
    main_dataset = dataset["main"]

    questions = main_dataset["question"]
    text_answers = main_dataset["answer"]

    number_answers = []
    for answer in text_answers:
        number_answers.append(int(answer.split("###")[1].strip()))

    final_data = [
        {"problem": question, "answer": answer}
        for question, answer in zip(questions, number_answers)
    ]

    with open(args.output_path, "w") as f:
        json.dump(final_data, f)

    random.shuffle(final_data)
    subset_data = final_data[: args.num_subset_examples]

    with open(args.subset_output_path, "w") as f:
        json.dump(subset_data, f)

    print(f"Saved {args.num_subset_examples} examples to {args.subset_output_path}")


if __name__ == "__main__":
    main()
