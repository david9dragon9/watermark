from typing import List
from pydantic import BaseModel
from openai import OpenAI
import argparse
import tqdm
import json


class TopicsList(BaseModel):
    topics: List[str]


class QuestionsList(BaseModel):
    creative_questions: List[str]
    explanation_questions: List[str]
    opinion_questions: List[str]
    miscellaneous_questions: List[str]


topics_prompt = "What are {num_topics} broad subjects that a user may ask an LLM chatbot about (e.g. sports, arts, chemistry, biology, religion)?"

questions_prompt = "What are some questions that a user may ask an LLM chatbot about {topic}? Generate {num_questions_per_topic} questions for each of the following categories relating to the topic: creative questions (e.g. write a story, create a poem), explanation questions (e.g. explain a concept, explain a process), opinion questions (e.g. what is your opinion on X, do you like X), and miscellaneous questions."


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate questions for a hierarchical chatbot"
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        required=True,
        help="Number of topics to generate questions for",
    )
    parser.add_argument(
        "--num_questions_per_topic",
        type=int,
        required=True,
        help="Number of questions to generate for each topic",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated questions",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name to use for generating questions",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each prompt",
    )
    return parser.parse_args()


def hierarchical_generate_questions(
    num_topics: int,
    num_questions_per_topic: int,
    output_path: str,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 512,
):
    try:
        from google.colab import userdata

        api_key = userdata.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
    except:
        client = OpenAI()
    assert output_path.endswith(".json"), "Output path must end with .json"

    topics = client.beta.chat.completions.parse(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        messages=[
            {"role": "user", "content": topics_prompt.format(num_topics=num_topics)}
        ],
        response_format=TopicsList,
    )

    topics_list = topics.choices[0].message.parsed.topics

    all_questions = []
    pbar = tqdm.tqdm(topics_list)
    for topic in pbar:
        pbar.set_description(f"Generating questions for {topic}")
        questions = client.beta.chat.completions.parse(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            messages=[
                {
                    "role": "user",
                    "content": questions_prompt.format(
                        topic=topic, num_questions_per_topic=num_questions_per_topic
                    ),
                }
            ],
            response_format=QuestionsList,
        )
        curr_parsed = questions.choices[0].message.parsed
        curr_questions = (
            curr_parsed.creative_questions
            + curr_parsed.explanation_questions
            + curr_parsed.opinion_questions
            + curr_parsed.miscellaneous_questions
        )
        all_questions.extend(curr_questions)

    with open(output_path, "w") as f:
        json.dump(all_questions, f)


def main():
    args = get_args()

    hierarchical_generate_questions(
        num_topics=args.num_topics,
        num_questions_per_topic=args.num_questions_per_topic,
        output_path=args.output_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
