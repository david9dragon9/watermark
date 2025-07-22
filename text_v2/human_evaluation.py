from typing import Optional
import argparse
import csv
import random
import tkinter as tk
from tkinter import ttk
import json


def get_args():
    parser = argparse.ArgumentParser(description="Text Comparison GUI")

    parser.add_argument(
        "--prompts", type=str, required=True, help="Path to the prompts file"
    )
    parser.add_argument(
        "--responses_1",
        type=str,
        required=True,
        help="Path to the first set of responses",
    )
    parser.add_argument(
        "--responses_2",
        type=str,
        required=True,
        help="Path to the second set of responses",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--subset_num",
        type=int,
        default=None,
        help="Number of pairs to label (optional)",
    )

    args = parser.parse_args()

    return args


def load_texts(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_labels(output_path, labels):
    with open(output_path, "w") as f:
        json.dump(labels, f)


class TextLabelingApp:
    def __init__(
        self,
        root,
        prompts,
        responses_1,
        responses_2,
        output_path,
        subset_num: Optional[int] = None,
    ):
        self.root = root
        self.prompts = prompts
        self.responses_1 = responses_1
        self.responses_2 = responses_2
        self.output_path = output_path
        self.index = 0
        self.labels = []
        self.total = len(prompts) if subset_num is None else subset_num

        self.setup_ui()
        self.load_next_pair()

    def setup_ui(self):
        self.root.title("Text Comparison")
        self.root.geometry("800x400")

        self.prompt_label = ttk.Label(
            self.root, text="", wraplength=780, font=("Arial", 12)
        )
        self.prompt_label.pack(pady=10)

        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.left_text = tk.Text(self.frame, wrap=tk.WORD, height=10, width=40)
        self.left_text.pack(side=tk.LEFT, padx=10)

        self.right_text = tk.Text(self.frame, wrap=tk.WORD, height=10, width=40)
        self.right_text.pack(side=tk.RIGHT, padx=10)

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.left_button = ttk.Button(
            self.button_frame, text="Choose Left", command=lambda: self.record_choice(0)
        )
        self.left_button.pack(side=tk.LEFT, padx=10)

        self.same_button = ttk.Button(
            self.button_frame, text="Same", command=lambda: self.record_choice(2)
        )
        self.same_button.pack(side=tk.LEFT, padx=10)

        self.right_button = ttk.Button(
            self.button_frame,
            text="Choose Right",
            command=lambda: self.record_choice(1),
        )
        self.right_button.pack(side=tk.RIGHT, padx=10)

    def load_next_pair(self):
        if self.index >= self.total:
            self.root.quit()
            return

        self.prompt_label.config(
            text=f"Prompt {self.index+1}: {self.prompts[self.index]}"
        )

        self.current_pair = [
            (self.responses_1[self.index], 1),
            (self.responses_2[self.index], 2),
        ]
        random.shuffle(self.current_pair)

        self.left_text.delete(1.0, tk.END)
        self.right_text.delete(1.0, tk.END)

        self.left_text.insert(tk.END, self.current_pair[0][0])
        self.right_text.insert(tk.END, self.current_pair[1][0])

    def record_choice(self, choice):
        if choice == 2:
            self.labels.append({"prompt": self.prompts[self.index], "chosen": "same"})
        else:
            self.labels.append(
                {
                    "prompt": self.prompts[self.index],
                    "chosen": self.current_pair[choice][1],
                }
            )

        save_labels(self.output_path, self.labels)
        self.index += 1
        self.load_next_pair()


def main():
    args = get_args()

    prompts = load_texts(args.prompts)
    responses_1 = load_texts(args.responses_1)
    responses_2 = load_texts(args.responses_2)

    if not (len(prompts) == len(responses_1) == len(responses_2)):
        raise ValueError("The number of prompts and responses must be the same")

    root = tk.Tk()
    app = TextLabelingApp(root, prompts, responses_1, responses_2, args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
