from typing import List, Dict, Union, Optional
from openai import OpenAI
import google.generativeai as genai
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import tqdm

base_url_mapping = {
    "sambanova": "https://api.sambanova.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "glhf": "https://glhf.chat/api/openai/v1",
    "scaleway": "https://api.scaleway.ai/v1",
}


@dataclass
class TokenStats:
    input_tokens: int
    output_tokens: int
    total_tokens: int


def create_client(service: str, api_key: str):
    if service in base_url_mapping:
        return OpenAI(base_url=base_url_mapping[service], api_key=api_key)
    elif service == "openai":
        return OpenAI(api_key=api_key)
    elif service == "gemini":
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash-exp")
    else:
        raise ValueError("Invalid service")


supported_models_mapping = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ],
    "sambanova": [
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.2-1B-Instruct",
        "Meta-Llama-3.2-3B-Instruct",
        "Meta-Llama-3.2-11B-Instruct",
        "Meta-Llama-3.2-90B-Instruct",
        "Meta-Llama-3.3-70B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen2.5-32B-Preview",
    ],
    "groq": [
        "gemma2-9b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-90b-vision-preview",
        "llama-3.3-70b-specdec",
        "llama-3.3-70b-versatile",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ],
    "together": ["Llama-Vision-Free"],
    "mistral": [
        "open-mistral-7b",
        "open-mistral-8x7b",
        "open-mistral-8x22b",
        "mistral-small-2402",
        "mistral-small-2409",
        "mistral-medium",
        "mistral-large-2402",
        "mistral-large-2407",
        "mistral-large-2411",
        "mistral-embed",
        "open-mistral-nemo",
        "ministral-3b-2410",
        "ministral-8b-2410",
        "mistral-moderation-2411",
    ],
    "glhf": [
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.2-1B-Instruct",
        "Meta-Llama-3.2-3B-Instruct",
        "Meta-Llama-3.2-11B-Instruct",
        "Meta-Llama-3.2-90B-Instruct",
        "Meta-Llama-3.3-70B-Instruct",
        "Qwen2.5-72B-Instruct",
    ],
    "scaleway": [
        "llama-3.1-8b-instruct",
        "llama-3.1-70b-instruct",
        "mistral-nemo-instruct-2407",
        "pixtral-12b-2409",
        "qwen2.5-32b-instruct",
    ],
}


def call_llm(
    messages: Union[List[Dict[str, str]], str],
    model: str,
    service: str,
    client: Union[OpenAI, genai.GenerativeModel],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    response_format: Optional[Dict[str, str]] = None,
):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if service == "openai" and response_format is not None:
        result = client.beta.chat.completions.parse(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        response_text = result.choices[0].message.content
        token_stats = TokenStats(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        )
        return response_text, token_stats
    elif service in base_url_mapping:
        result = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = result.choices[0].message.content
        token_stats = TokenStats(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        )
        return response_text, token_stats
    elif service == "gemini":
        assert len(messages) == 1
        message = messages[0]["content"]
        result = model.generate_content(message)

        response_text = result.text
        token_stats = TokenStats(
            input_tokens=result.usage_metadata.prompt_token_count,
            output_tokens=result.usage_metadata.candidates_token_count,
            total_tokens=result.usage_metadata.total_token_count,
        )
        return response_text, token_stats
    else:
        raise ValueError("Invalid service")


def bold(text: str):
    return f"\033[1m{text}\033[0m"


def get_supported_models(service: str):
    print(bold(f"Supported models for {service}"))
    for model in supported_models_mapping[service]:
        print(model)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def batch_call(
    batch_messages: List[List[Dict[str, str]]],
    model: str,
    service: str,
    client: Union[OpenAI, genai.GenerativeModel],
    rpt: int = 1000000,
    t: int = 60.1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    response_format: Optional[Dict[str, str]] = None,
):
    def batch_completion(curr_messages):
        kwargs = {
            "model": model,
            "service": service,
            "client": client,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
        }
        completions = []
        with ThreadPoolExecutor(max_workers=len(curr_messages)) as executor:
            for sub_batch in chunks(curr_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = kwargs.copy()
                    kwargs_modified["messages"] = message_list
                    future = executor.submit(call_llm, **kwargs_modified)
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

    pbar = tqdm.tqdm(total=len(batch_messages))
    all_completions = []
    curr_index = 0
    for batch in chunks(batch_messages, rpt):
        start = time.time()
        completions = batch_completion(batch)
        end = time.time()
        pbar.set_description(f"Batch completed in {end - start:.2f}s")

        if curr_index + rpt < len(batch_messages):
            time.sleep(max(0, t - (end - start)))
        curr_index += rpt

        pbar.update(len(batch))
        all_completions.extend(completions)
    pbar.close()
    return all_completions
