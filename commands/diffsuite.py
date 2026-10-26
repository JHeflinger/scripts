"""
Visualize progressive PNG renders and their pixel-wise differences.

Usage:
    python png_progressive_diff.py ground_truth.png [progressive1.png ... progressiveN.png]

Dependencies:
    pip install pillow numpy
"""

import argparse
import math
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk

ORANGE = np.array([255.0, 112.0, 0.0], dtype=np.float32)

def load_rgb(path):
    """Load a PNG as RGB, compositing transparency over black."""
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def mse(a, b):
    """Mean squared RGB error, averaged over pixels and color channels."""
    if a.shape != b.shape:
        raise ValueError(f"Image dimensions differ: {a.shape[1]}x{a.shape[0]} vs "
                         f"{b.shape[1]}x{b.shape[0]}")
    delta = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(delta * delta))


def make_diff(a, b):
    """
    Return an RGB diff image and MSE(a, b).
    Pixel brightness is proportional to that pixel's mean squared RGB error:
    sqrt(error / max_error) is used for more visible low-level differences.
    """
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    delta = af - bf
    per_pixel_mse = np.mean(delta * delta, axis=2)
    global_mse = float(np.mean(per_pixel_mse))

    # A pixel's maximum possible mean squared RGB error is 255**2.
    # Square-root mapping makes subtle differences easier to see while
    # retaining a monotonic relationship between error and brightness.
    intensity = np.sqrt(np.clip(per_pixel_mse / (255.0 ** 2), 0.0, 1.0))
    out = np.zeros((*per_pixel_mse.shape, 3), dtype=np.uint8)
    out[:] = 0
    out_float = ORANGE[None, None, :] * intensity[..., None]
    out[:] = np.clip(out_float, 0, 255).astype(np.uint8)
    return out, global_mse


def format_mse(value):
    return f"MSE: {value:.6g}"


class DiffViewer:
    def __init__(self, root, paths, images):
        self.root = root
        self.paths = paths
        self.images = images
        self.photo_refs = []
        self.max_image_width = 360
        self.max_image_height = 280

        root.title("Progressive PNG Difference Viewer")
        root.geometry("1400x850")

        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        hscroll = ttk.Scrollbar(outer, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self.content = ttk.Frame(self.canvas, padding=10)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

        self._build_grid()

    def _update_scrollregion(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-event.delta / 120), "units")

    def _fit(self, array):
        image = Image.fromarray(array, mode="RGB")
        image.thumbnail((self.max_image_width, self.max_image_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _add_image(self, parent, array, caption, metric=None):
        frame = ttk.Frame(parent, padding=6, relief="groove")
        frame.pack(side="left", anchor="n", padx=3, pady=3)
        ttk.Label(frame, text=caption, wraplength=self.max_image_width,
                  justify="center").pack(fill="x", pady=(0, 5))
        photo = self._fit(array)
        self.photo_refs.append(photo)
        ttk.Label(frame, image=photo).pack()
        if metric is not None:
            ttk.Label(frame, text=format_mse(metric)).pack(pady=(5, 0))

    def _build_grid(self):
        gt = self.images[0]

        header = ttk.Frame(self.content)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text=f"Ground truth: {self.paths[0]}",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(header, text=f"Resolution: {gt.shape[1]} × {gt.shape[0]}").pack(anchor="w")
        ttk.Label(
            header,
            text="Each column is an input image in argument order. Diffs use black for no error "
                 "and orange brightness for increasing per-pixel RGB error. MSE is averaged "
                 "across all pixels and RGB channels."
        ).pack(anchor="w")

        # One column per input image, in the same order as the command-line arguments.
        grid = ttk.Frame(self.content)
        grid.pack(anchor="nw")

        for i, (path, current) in enumerate(zip(self.paths, self.images)):
            column = ttk.LabelFrame(
                grid,
                text=("Ground truth" if i == 0 else f"Progressive {i}") +
                     f"\n{os.path.basename(path)}",
                padding=5,
            )
            column.grid(row=0, column=i, sticky="n", padx=4, pady=4)

            self._add_image(column, current, "Input image")

            if i == 0:
                # Ground truth compared with itself has zero error; no preceding image exists.
                zero_diff = np.zeros_like(gt)
                self._add_image(column, zero_diff, "Difference from previous image\n(no previous image)")
                self._add_image(column, zero_diff, "Difference from ground truth", 0.0)
            else:
                prev_diff, prev_mse = make_diff(current, self.images[i - 1])
                gt_diff, gt_mse = make_diff(current, gt)
                self._add_image(column, prev_diff, "Difference from previous image", prev_mse)
                self._add_image(column, gt_diff, "Difference from ground truth", gt_mse)


def main():
    parser = argparse.ArgumentParser(
        description="Display progressive PNGs alongside previous-image and ground-truth diffs."
    )
    parser.add_argument("pngs", nargs="+", help="Ground-truth PNG first, then progressive PNGs")
    args = parser.parse_args()

    if len(args.pngs) < 2:
        parser.error("provide a ground-truth PNG followed by at least one progressive PNG")

    try:
        images = [load_rgb(path) for path in args.pngs]
        reference_shape = images[0].shape
        for path, image in zip(args.pngs[1:], images[1:]):
            if image.shape != reference_shape:
                raise ValueError(
                    f"{path} is {image.shape[1]}x{image.shape[0]}, while ground truth is "
                    f"{reference_shape[1]}x{reference_shape[0]}. All images must have "
                    "identical dimensions."
                )

        root = tk.Tk()
        DiffViewer(root, args.pngs, images)
        root.mainloop()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
