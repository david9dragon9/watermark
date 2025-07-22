from typing import Optional, List

import os
import json
import math
import argparse
import torch.multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--descriptions_path", type=str, required=True)
    parser.add_argument("--outputs_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_return_sequences", type=int, default=1)

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--system_message", type=str, default=None)

    return parser.parse_args()


def main():
    args = get_args()

    assert args.descriptions_path.endswith(".json")
    with open(args.descriptions_path, "r") as f:
        descriptions = json.load(f)

    chunk_size = math.ceil(len(descriptions) / args.num_gpus)
    chunks = [
        descriptions[i * chunk_size : (i + 1) * chunk_size]
        for i in range(args.num_gpus)
    ]

    description_folder, description_filename = os.path.split(args.descriptions_path)
    description_filename = description_filename[:-5]
    outputs_folder, outputs_filename = os.path.split(args.outputs_path)
    outputs_filename = outputs_filename[:-5]

    all_cmds = []
    all_output_paths = []
    if args.system_message is not None:
        args.system_message = f'"{args.system_message}"'
    for gpu_num in range(args.num_gpus):
        curr_description_path = os.path.join(
            description_folder, f"{description_filename}_{gpu_num}.json"
        )
        with open(curr_description_path, "w") as f:
            json.dump(chunks[gpu_num], f)

        curr_outputs_path = os.path.join(
            outputs_folder, f"{outputs_filename}_{gpu_num}.json"
        )
        all_output_paths.append(curr_outputs_path)
        cmd = f"""CUDA_VISIBLE_DEVICES={gpu_num} python batch_inference.py \
                    --descriptions_path {curr_description_path} \
                    --outputs_path {curr_outputs_path} \
                    --model_name {args.model_name} \
                    --max_new_tokens {args.max_new_tokens} \
                    --temperature {args.temperature} \
                    --num_return_sequences {args.num_return_sequences} \
                    --gpu_memory_utilization {args.gpu_memory_utilization} \
                    --system_message {args.system_message}"""
        all_cmds.append(cmd)

    processes = []
    for cmd in all_cmds:
        processes.append(mp.Process(target=os.system, args=(cmd,)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    results = []
    for output_path in all_output_paths:
        with open(output_path, "r") as f:
            results.extend(json.load(f))

    with open(args.outputs_path, "w") as f:
        json.dump(results, f)

    print(f"Outputs saved to {args.outputs_path}")


if __name__ == "__main__":
    main()
