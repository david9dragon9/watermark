import argparse
import os
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json


def get_args():
    parser = argparse.ArgumentParser(description="Image Pair Labeling Tool")

    parser.add_argument("--folder1", type=str, help="Path to first image folder")
    parser.add_argument("--folder2", type=str, help="Path to second image folder")
    parser.add_argument("--output", type=str, help="Path to output JSON file")

    args = parser.parse_args()

    return args


def get_image_pairs(folder1, folder2):
    images1 = {
        os.path.splitext(f)[0]: os.path.join(folder1, f)
        for f in os.listdir(folder1)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    }
    images2 = {
        os.path.splitext(f)[0]: os.path.join(folder2, f)
        for f in os.listdir(folder2)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    }
    common_names = set(images1.keys()) & set(images2.keys())
    return [(images1[name], images2[name]) for name in common_names]


class ImageLabeler:
    def __init__(self, root, image_pairs, output_path):
        self.root = root
        self.image_pairs = image_pairs
        self.output_path = output_path
        self.results = []
        self.current_index = 0

        self.root.title("Image Comparison Tool")
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.left_label = ttk.Label(self.frame)
        self.left_label.grid(row=0, column=0, padx=10, pady=10)

        self.right_label = ttk.Label(self.frame)
        self.right_label.grid(row=0, column=1, padx=10, pady=10)

        self.button_left = ttk.Button(
            self.frame,
            text="Choose Left",
            command=lambda: self.save_choice(self.left_folder),
        )
        self.button_left.grid(row=1, column=0, padx=10, pady=10)

        self.button_right = ttk.Button(
            self.frame,
            text="Choose Right",
            command=lambda: self.save_choice(self.right_folder),
        )
        self.button_right.grid(row=1, column=1, padx=10, pady=10)

        self.load_next_image()

    def load_next_image(self):
        if self.current_index >= len(self.image_pairs):
            self.root.quit()
            return

        img1_path, img2_path = self.image_pairs[self.current_index]
        pair = [(img1_path, 1), (img2_path, 2)]
        random.shuffle(pair)
        (self.left_path, self.left_folder), (self.right_path, self.right_folder) = pair

        self.display_image(self.left_path, self.left_label)
        self.display_image(self.right_path, self.right_label)

    def display_image(self, img_path, label):
        img = Image.open(img_path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        label.img = img
        label.config(image=img)

    def save_choice(self, chosen_folder):
        self.results.append(
            {"pair_index": self.current_index, "chosen_folder": chosen_folder}
        )
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        self.current_index += 1
        self.load_next_image()


def main():
    args = get_args()

    image_pairs = get_image_pairs(args.folder1, args.folder2)
    if not image_pairs:
        print("No matching image pairs found.")
        exit()

    root = tk.Tk()
    app = ImageLabeler(root, image_pairs, args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
