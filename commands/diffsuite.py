"""
Visualize progressive PNG renders and their pixel-wise differences.

Usage:
    python png_progressive_diff.py ground_truth.png [progressive1.png ... progressiveN.png]

Dependencies:
    pip install pillow numpy
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


ORANGE = np.array([255.0, 112.0, 0.0], dtype=np.float32)


def load_rgb(path):
    """Load an image as RGB, compositing transparency over black."""
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def make_diff(a, b, brightness=1.0):
    """
    Return an orange-on-black difference image and its RGB mean squared error.

    Brightness changes only the visualization, not the MSE.
    """
    if a.shape != b.shape:
        raise ValueError("Cannot compare images with different dimensions.")

    delta = a.astype(np.float32) - b.astype(np.float32)
    per_pixel_mse = np.mean(delta * delta, axis=2)
    global_mse = float(np.mean(per_pixel_mse))

    # Square-root scaling makes subtle differences easier to see.
    intensity = np.sqrt(
        np.clip(per_pixel_mse / (255.0 ** 2), 0.0, 1.0)
    )
    intensity = np.clip(intensity * brightness, 0.0, 1.0)

    diff = (
        intensity[..., None] * ORANGE[None, None, :]
    ).astype(np.uint8)

    return diff, global_mse


def format_mse(value):
    return f"MSE: {value:.6g}"


class DiffViewer:
    def __init__(self, root, paths, images):
        self.root = root
        self.paths = paths
        self.images = images

        self.photo_refs = []
        self.base_image_width = 360
        self.base_image_height = 280

        self.brightness = tk.DoubleVar(value=1.0)
        self.zoom = tk.DoubleVar(value=1.0)

        root.title("Progressive PNG Difference Viewer")
        root.geometry("1200x850")
        root.minsize(600, 400)

        # Controls
        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Orange brightness").pack(side="left")

        ttk.Scale(
            toolbar,
            from_=0.0,
            to=4.0,
            variable=self.brightness,
            orient="horizontal",
            length=150,
            command=lambda _v: self._refresh(),
        ).pack(side="left", padx=(6, 16))

        self.brightness_label = ttk.Label(toolbar, text="1.00x")
        self.brightness_label.pack(side="left", padx=(0, 20))

        ttk.Label(toolbar, text="Zoom").pack(side="left")

        ttk.Scale(
            toolbar,
            from_=0.25,
            to=3.0,
            variable=self.zoom,
            orient="horizontal",
            length=150,
            command=lambda _v: self._refresh(),
        ).pack(side="left", padx=6)

        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side="left", padx=6)

        ttk.Button(
            toolbar,
            text="Reset",
            command=self._reset_controls,
        ).pack(side="left", padx=12)

        # Scrollable area
        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(
            outer, orient="vertical", command=self.canvas.yview
        )
        hscroll = ttk.Scrollbar(
            outer, orient="horizontal", command=self.canvas.xview
        )

        self.canvas.configure(
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")

        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self.content = ttk.Frame(self.canvas, padding=10)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )

        self.content.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind_all("<Control-MouseWheel>", self._on_zoom_wheel)

        # Linux wheel support
        self.canvas.bind_all("<Button-4>", self._on_linux_wheel_up)
        self.canvas.bind_all("<Button-5>", self._on_linux_wheel_down)

        self._build_grid()

    def _reset_controls(self):
        self.brightness.set(1.0)
        self.zoom.set(1.0)
        self._refresh()

    def _update_scrollregion(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(
                -int(event.delta / 120) or (-1 if event.delta > 0 else 1),
                "units",
            )

    def _on_shift_mousewheel(self, event):
        if event.delta:
            self.canvas.xview_scroll(
                -int(event.delta / 120) or (-1 if event.delta > 0 else 1),
                "units",
            )

    def _on_zoom_wheel(self, event):
        step = 0.1 if event.delta > 0 else -0.1
        self.zoom.set(
            max(0.25, min(3.0, self.zoom.get() + step))
        )
        self._refresh()

    def _on_linux_wheel_up(self, event):
        if event.state & 0x0004:  # Ctrl held
            self.zoom.set(min(3.0, self.zoom.get() + 0.1))
            self._refresh()
        else:
            self.canvas.yview_scroll(-1, "units")

    def _on_linux_wheel_down(self, event):
        if event.state & 0x0004:  # Ctrl held
            self.zoom.set(max(0.25, self.zoom.get() - 0.1))
            self._refresh()
        else:
            self.canvas.yview_scroll(1, "units")

    def _fit(self, array):
        image = Image.fromarray(array, mode="RGB")

        max_width = max(
            1, int(self.base_image_width * self.zoom.get())
        )
        max_height = max(
            1, int(self.base_image_height * self.zoom.get())
        )

        image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS,
        )

        return ImageTk.PhotoImage(image)


    def _add_image(self, parent, array, caption, metric=None):
        frame = ttk.Frame(parent, padding=5, relief="groove")
        frame.pack(side="left", anchor="n", padx=3, pady=3)

        width = max(150, int(self.base_image_width * self.zoom.get()))

        ttk.Label(
            frame,
            text=caption,
            wraplength=width,
            justify="center",
        ).pack(fill="x", pady=(0, 5))

        photo = self._fit(array)
        self.photo_refs.append(photo)
        ttk.Label(frame, image=photo).pack()

        if metric is not None:
            ttk.Label(frame, text=format_mse(metric)).pack(pady=(5, 0))

    def _build_grid(self):
        gt = self.images[0]
    
        header = ttk.Frame(self.content)
        header.pack(fill="x", pady=(0, 10))
    
        ttk.Label(
            header,
            text=f"Ground truth: {self.paths[0]}",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")
    
        ttk.Label(
            header,
            text=f"Resolution: {gt.shape[1]} x {gt.shape[0]}",
        ).pack(anchor="w")
    
        for i, (path, current) in enumerate(zip(self.paths, self.images)):
            title = "Ground truth" if i == 0 else f"Progressive {i}"
    
            section = ttk.LabelFrame(
                self.content,
                text=f"{title} - {os.path.basename(path)}",
                padding=5,
            )
            section.pack(fill="x", anchor="w", pady=8)
    
            # Original image on the left.
            self._add_image(section, current, "Input image")
    
            if i == 0:
                zero_diff = np.zeros_like(gt)
    
                self._add_image(
                    section, zero_diff,
                    "Difference from previous image (N/A)",
                )
                self._add_image(
                    section, zero_diff,
                    "Difference from ground truth", 0.0,
                )
                continue
    
            prev_diff, prev_mse = make_diff(
                current, self.images[i - 1], self.brightness.get()
            )
            gt_diff, gt_mse = make_diff(
                current, gt, self.brightness.get()
            )
    
            # Both diffs appear to the right of the original.
            self._add_image(
                section, prev_diff,
                "Difference from previous image", prev_mse,
            )
            self._add_image(
                section, gt_diff,
                "Difference from ground truth", gt_mse,
            )
    
    def _refresh(self):
        self.brightness_label.configure(
            text=f"{self.brightness.get():.2f}x"
        )
        self.zoom_label.configure(
            text=f"{self.zoom.get() * 100:.0f}%"
        )

        # Rebuild the widgets so the images reflect the new settings.
        self.photo_refs.clear()

        for child in self.content.winfo_children():
            child.destroy()

        self._build_grid()
        self._update_scrollregion()

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Display progressive PNGs with previous-image and "
            "ground-truth difference images."
        )
    )

    parser.add_argument(
        "pngs",
        nargs="+",
        help="Ground-truth PNG first, followed by progressive PNGs",
    )

    args = parser.parse_args()

    if len(args.pngs) < 2:
        parser.error(
            "Provide a ground-truth PNG and at least one progressive PNG."
        )

    try:
        images = [load_rgb(path) for path in args.pngs]

        reference_shape = images[0].shape

        for path, image in zip(args.pngs[1:], images[1:]):
            if image.shape != reference_shape:
                raise ValueError(
                    f"{path}: dimensions {image.shape[1]}x{image.shape[0]} "
                    f"do not match ground truth "
                    f"{reference_shape[1]}x{reference_shape[0]}."
                )

        root = tk.Tk()
        DiffViewer(root, args.pngs, images)
        root.mainloop()

    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
