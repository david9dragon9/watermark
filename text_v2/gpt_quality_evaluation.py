from typing import List, Dict, Optional
from pydantic import BaseModel
import openai
from openai import OpenAI
import json
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import os


class GPTQualityEvaluation(BaseModel):
    short_reasoning: str
    choice: str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question_input_path", type=str, help="Path to the input question file"
    )
    parser.add_argument(
        "--response1_input_path", type=str, help="Path to the input response file 1"
    )
    parser.add_argument(
        "--response2_input_path", type=str, help="Path to the input response file 2"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to save the statistics and figures"
    )

    return parser.parse_args()


evaluation_prompt = "Determine which of the responses to the question {question} is higher quality (in terms of content, grammar, and coherence). If a response is cut off, do not penalize it for the cut-off, merely evaluate the content that is present. Output a short (1-2 sentence) reasoning before making a final choice of letter (A or B).\nResponse A: {response1}\nResponse B: {response2}\n"


def call_openai_llm(
    messages: List[Dict[str, str]], client: openai.OpenAI, max_tokens: int = 512
):
    response = client.beta.chat.completions.parse(
        messages=messages,
        model="gpt-4o-mini",
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        response_format=GPTQualityEvaluation,
    )

    return response.choices[0].message.parsed.choice


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def batch_openai_completion(curr_messages, client, max_tokens):
    kwargs = {
        "client": client,
        "max_tokens": max_tokens,
    }
    completions = []
    with ThreadPoolExecutor(max_workers=len(curr_messages)) as executor:
        for sub_batch in chunks(curr_messages, 100):
            for message_list in sub_batch:
                kwargs_modified = kwargs.copy()
                kwargs_modified["messages"] = message_list
                future = executor.submit(call_openai_llm, **kwargs_modified)
                completions.append(future)

    # Retrieve the results from the futures
    # results = [future.result() for future in completions]
    # return exceptions if any
    results = []
    for future in completions:
        try:
            results.append(future.result())
        except Exception as exc:
            results.append(exc)
    return results


def evaluate_quality(
    questions: List[str],
    responses1: List[str],
    responses2: List[str],
    client: openai.OpenAI,
    evaluation_prompt: str = evaluation_prompt,
    actual_choices: Optional[List[int]] = None,
    use_question: bool = True,
):

    openai_prompts = []
    use_actual_choices = True if actual_choices is None else False
    actual_choices = [] if actual_choices is None else actual_choices
    for question, response1, response2 in zip(questions, responses1, responses2):
        choice_a = 1 if random.random() > 0.5 else 2
        if use_actual_choices:
            actual_choices.append(choice_a)
        response_a = response1 if choice_a == 1 else response2
        response_b = response2 if choice_a == 1 else response1
        if use_question:
            curr_prompt = evaluation_prompt.format(
                question=question, response1=response_a, response2=response_b
            )
        else:
            curr_prompt = evaluation_prompt.format(
                response1=response_a, response2=response_b
            )

        openai_prompts.append(curr_prompt)

    openai_responses = batch_openai_completion(openai_prompts, client, 512)

    actual_chose_1 = 0
    for actual_choice, openai_response in zip(actual_choices, openai_responses):
        if openai_response == "A" and actual_choice == 1:
            actual_chose_1 += 1
        elif openai_response == "B" and actual_choice == 2:
            actual_chose_1 += 1

    accuracy = actual_chose_1 / len(actual_choices)

    return accuracy, actual_choices, openai_responses


def main():
    client = OpenAI()

    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.question_input_path, "r") as f:
        questions = json.load(f)

    with open(args.response1_input_path, "r") as f:
        responses1 = json.load(f)

    with open(args.response2_input_path, "r") as f:
        responses2 = json.load(f)

    accuracy, actual_choices, openai_responses = evaluate_quality(
        questions=questions,
        responses1=responses1,
        responses2=responses2,
        client=client,
    )

    all_statistics = {
        "accuracy": accuracy,
        "actual_choices": actual_choices,
        "openai_responses": openai_responses,
    }

    with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
        json.dump(all_statistics, f)
