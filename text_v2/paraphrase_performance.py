from openai import OpenAI
import json
from text_v2.vllm_watermark import vllm_watermark_generate, vllm_watermark_detect
import argparse
from text_v2.gpt_quality_evaluation import evaluate_quality


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts", type=str, help="Path to the input prompt file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save the generated text"
    )

    return parser.parse_args()


def main():
    args = get_args()

    client = OpenAI()

    with open(args.prompts, "r") as f:
        prompts = json.load(f)

    # Generate the text with watermark
    normal_responses = vllm_watermark_generate(
        prompts,
        password="capstone",
        output_path=args.output_path,
        use_watermark=False,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
    )

    # Generate the text with watermark
    duplicate_responses = vllm_watermark_generate(
        prompts,
        password="capstone",
        output_path=args.output_path,
        use_watermark=False,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        temperature=1.0,
    )

    paraphrase_prompts = [
        f"Paraphrase the following text. Do not change the meaning of thee text, but simply substitute synonyms that have the same meaning. Only output the paraphrased text and NOTHING else. Here is the text to paraphrase: {response}"
        for response in normal_responses
    ]

    # Generate the text with watermark
    paraphrased_responses = vllm_watermark_generate(
        paraphrase_prompts,
        password="capstone",
        output_path=args.output_path,
        use_watermark=True,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
    )

    difference_prompt = "Compare the following two texts and output a final choice of whether the two texts differ substantially in meaning. Only output 'YES' if the two texts differ substantially in meaning, and 'NO' if they do not. Here are the two texts: {response1} and {response2}"
    _, diff_actual_choices, diff_openai_explanations = evaluate_quality(
        questions=["DUMMY PROMPT"] * len(paraphrased_responses),
        responses1=normal_responses,
        responses2=paraphrased_responses,
        client=client,
        evaluation_prompt=difference_prompt,
        actual_choices=None,
        use_question=False,
    )
    diff_accuracy_yes = sum(int(x == "YES") for x in diff_actual_choices) / len(
        diff_actual_choices
    )

    _, shift_actual_choices, shift_openai_explanations = evaluate_quality(
        questions=["DUMMY PROMPT"] * len(paraphrased_responses),
        responses1=normal_responses,
        responses2=normal_responses[1:] + [normal_responses[0]],
        client=client,
        evaluation_prompt=difference_prompt,
        actual_choices=None,
        use_question=False,
    )
    shift_accuracy_yes = sum(int(x == "YES") for x in shift_actual_choices) / len(
        shift_actual_choices
    )

    _, same_actual_choices, same_openai_explanations = evaluate_quality(
        questions=["DUMMY PROMPT"] * len(paraphrased_responses),
        responses1=normal_responses,
        responses2=duplicate_responses,
        client=client,
        evaluation_prompt=difference_prompt,
        actual_choices=None,
        use_question=False,
    )

    same_accuracy_yes = sum(int(x == "YES") for x in same_actual_choices) / len(
        same_actual_choices
    )

    print("Difference Accuracy YES:", diff_accuracy_yes)
    print("Shift Accuracy YES:", shift_accuracy_yes)
    print("Same Accuracy YES:", same_accuracy_yes)

    # Detect the watermark
    watermark_detections = vllm_watermark_detect(
        paraphrased_responses,
        password="capstone",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        score_length=128256,
    )

    vanilla_detections = vllm_watermark_detect(
        normal_responses,
        password="capstone",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        score_length=128256,
    )

    duplicate_detections = vllm_watermark_detect(
        duplicate_responses,
        password="capstone",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        hash_token_window=4,
        score_length=128256,
    )

    all_results = {
        "watermark_detections": watermark_detections,
        "vanilla_detections": vanilla_detections,
        "duplicate_detections": duplicate_detections,
        "diff_actual_choices": diff_actual_choices,
        "diff_openai_explanations": diff_openai_explanations,
        "shift_actual_choices": shift_actual_choices,
        "shift_openai_explanations": shift_openai_explanations,
        "same_actual_choices": same_actual_choices,
        "same_openai_explanations": same_openai_explanations,
        "diff_accuracy_yes": diff_accuracy_yes,
        "shift_accuracy_yes": shift_accuracy_yes,
        "same_accuracy_yes": same_accuracy_yes,
    }

    with open(args.output_path, "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    main()
