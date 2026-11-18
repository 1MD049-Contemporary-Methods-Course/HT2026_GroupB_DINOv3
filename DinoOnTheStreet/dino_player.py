import argparse
import pickle
import time
import tkinter as tk
from tkinter import ttk, messagebox
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision.transforms import functional as TF

from interactive_dino import resolve_image, draw_boxes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fullscreen interactive DINOv3 video explorer."
    )
    parser.add_argument("--repo", type=Path, default=Path("dinov3"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("dinov3_vits16.pth"),
    )
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--camera", default="CAM_F0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--cache", type=int, default=32)
    return parser.parse_args()


def format_time(seconds):
    minutes, seconds = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


class DinoPlayer:
    def __init__(self, window, args, model, infos, device):
        self.window = window
        self.args = args
        self.root_path = args.root.resolve()
        self.model = model
        self.infos = infos
        self.device = device

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.pending_key = None
        self.cache = OrderedDict()
        self.failed_keys = set()

        self.index = 0
        self.loaded_key = None
        self.image = None
        self.features = None
        self.original_size = None

        self.query = None
        self.query_source = None
        self.query_patch = None
        self.query_preview = None

        self.last_display = None
        self.display_geometry = None
        self.photo = None

        self.playing = False
        self.dragging = False
        self.closed = False
        self.fullscreen = True

        self.seek_job = None
        self.resize_job = None
        self.tick_job = None
        self.last_advance = time.monotonic()

        self.score_key = None
        self.score_map = None

        camera_sets = [set(info["cams"]) for info in infos]
        cameras = sorted(set.intersection(*camera_sets))

        if not cameras:
            raise ValueError("No shared camera streams found.")

        initial_camera = (
            args.camera if args.camera in cameras else cameras[0]
        )

        self.camera = tk.StringVar(value=initial_camera)
        self.boxes = tk.BooleanVar(value=True)
        self.floor = tk.DoubleVar(value=30.0)
        self.timeline = tk.DoubleVar(value=0)
        self.fps = tk.DoubleVar(value=max(0.1, args.fps))

        self.status = tk.StringVar(value="Loading first frame...")
        self.time_text = tk.StringVar()
        self.query_text = tk.StringVar(
            value="Click the left image to select a feature."
        )

        self.window.title("DINOv3 interactive video player")
        self.window.geometry("1280x850")
        self.window.minsize(850, 600)
        self.window.attributes("-fullscreen", True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.window.bind("<F11>", self.toggle_fullscreen)
        self.window.bind("<Escape>", self.exit_fullscreen)

        self.build_interface(cameras)

        self.update_time()
        self.request_frame()
        self.tick_job = self.window.after(30, self.tick)

    def build_interface(self, cameras):
        toolbar = ttk.Frame(self.window, padding=8)
        toolbar.pack(side="top", fill="x")

        ttk.Label(
            toolbar, text="Video stream:"
        ).pack(side="left")

        picker = ttk.Combobox(
            toolbar,
            textvariable=self.camera,
            values=cameras,
            state="readonly",
            width=12,
        )
        picker.pack(side="left", padx=8)
        picker.bind(
            "<<ComboboxSelected>>", self.change_camera
        )

        ttk.Checkbutton(
            toolbar,
            text="Annotation boxes",
            variable=self.boxes,
            command=self.render,
        ).pack(side="left", padx=8)

        ttk.Button(
            toolbar,
            text="Clear selection",
            command=self.clear_query,
        ).pack(side="left", padx=4)

        ttk.Button(
            toolbar,
            text="Save screenshot",
            command=self.save,
        ).pack(side="left", padx=4)

        ttk.Button(
            toolbar,
            text="Fullscreen (F11)",
            command=self.toggle_fullscreen,
        ).pack(side="right", padx=4)

        # Reserve the control area before allowing the viewer to expand.
        bottom = ttk.Frame(self.window, padding=8)
        bottom.pack(side="bottom", fill="x")

        ttk.Label(
            bottom,
            textvariable=self.status,
            anchor="w",
        ).pack(side="bottom", fill="x", pady=(4, 0))

        ttk.Label(
            bottom,
            textvariable=self.query_text,
            anchor="w",
        ).pack(side="bottom", fill="x", pady=(4, 0))

        similarity_controls = ttk.Frame(bottom)
        similarity_controls.pack(
            side="bottom", fill="x", pady=6
        )

        ttk.Label(
            similarity_controls,
            text="Similarity display floor:",
        ).pack(side="left")

        ttk.Scale(
            similarity_controls,
            from_=0,
            to=95,
            variable=self.floor,
            command=lambda value: self.render(),
        ).pack(
            side="left", fill="x", expand=True, padx=10
        )

        self.floor_label = ttk.Label(
            similarity_controls, width=6
        )
        self.floor_label.pack(side="left")

        self.seek = ttk.Scale(
            bottom,
            from_=0,
            to=max(1, len(self.infos) - 1),
            variable=self.timeline,
            command=self.seek_changed,
        )
        self.seek.pack(side="bottom", fill="x", pady=6)
        self.seek.bind("<ButtonPress-1>", self.seek_start)
        self.seek.bind("<ButtonRelease-1>", self.seek_end)

        playback_controls = ttk.Frame(bottom)
        playback_controls.pack(
            side="bottom", fill="x", pady=4
        )

        ttk.Button(
            playback_controls,
            text="Previous",
            command=lambda: self.step(-1),
        ).pack(side="left")

        self.play_button = ttk.Button(
            playback_controls,
            text="Play",
            command=self.toggle_play,
        )
        self.play_button.pack(side="left", padx=6)

        ttk.Button(
            playback_controls,
            text="Next",
            command=lambda: self.step(1),
        ).pack(side="left")

        ttk.Label(
            playback_controls, text="Target FPS:"
        ).pack(side="left", padx=(18, 5))

        ttk.Spinbox(
            playback_controls,
            from_=0.1,
            to=30,
            increment=0.5,
            textvariable=self.fps,
            width=5,
        ).pack(side="left")

        ttk.Label(
            playback_controls,
            textvariable=self.time_text,
        ).pack(side="right")

        titles = ttk.Frame(self.window)
        titles.pack(side="top", fill="x")
        titles.columnconfigure(0, weight=1)
        titles.columnconfigure(1, weight=1)

        ttk.Label(
            titles,
            text="Camera — click to select; playback continues",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", pady=3)

        ttk.Label(
            titles,
            text="DINOv3 similarity — purple low / yellow high",
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", pady=3)

        self.viewer = tk.Canvas(
            self.window,
            background="#171717",
            highlightthickness=0,
            width=self.args.width * 2,
            height=int(self.args.width * 0.6),
        )
        self.viewer.pack(
            side="top",
            padx=8,
            pady=4,
            fill="both",
            expand=True,
        )

        self.canvas_image = self.viewer.create_image(
            0, 0, anchor="nw"
        )

        self.viewer.bind("<Button-1>", self.click_image)
        self.viewer.bind(
            "<Configure>", self.schedule_resize
        )

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.window.attributes(
            "-fullscreen", self.fullscreen
        )
        return "break"

    def exit_fullscreen(self, event=None):
        self.fullscreen = False
        self.window.attributes("-fullscreen", False)
        return "break"

    def current_key(self):
        return self.camera.get(), self.index

    def update_time(self):
        start = self.infos[0]["timestamp"]
        current = (
            self.infos[self.index]["timestamp"] - start
        ) / 1e6
        total = (
            self.infos[-1]["timestamp"] - start
        ) / 1e6

        self.time_text.set(
            f"{format_time(current)} / {format_time(total)}"
            f"    Frame {self.index + 1}/{len(self.infos)}"
        )

    def load_frame(self, key):
        camera_name, index = key

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        camera = self.infos[index]["cams"][camera_name]
        image_path = resolve_image(
            self.root_path, camera["data_path"]
        )

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        original_size = image.size
        original_width, original_height = original_size

        width = max(
            16, round(self.args.width / 16) * 16
        )
        height = max(
            16,
            round(
                original_height / original_width
                * width / 16
            ) * 16,
        )

        image = image.resize(
            (width, height),
            Image.Resampling.BICUBIC,
        )

        tensor = TF.normalize(
            TF.to_tensor(image),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = (
                self.model.get_intermediate_layers(
                    tensor,
                    n=1,
                    reshape=True,
                    norm=True,
                )[0][0]
            )
            features = F.normalize(
                features.float(), dim=0
            ).cpu()

        bgr = cv2.cvtColor(
            np.asarray(image), cv2.COLOR_RGB2BGR
        )
        result = bgr, features, original_size

        self.cache[key] = result
        while len(self.cache) > max(1, self.args.cache):
            self.cache.popitem(last=False)

        return result

    def request_frame(self):
        self.update_time()
        key = self.current_key()

        if key == self.loaded_key:
            self.status.set(
                f"Ready — {key[0]}, frame {key[1] + 1}"
            )
            self.render()
            return

        if key in self.failed_keys:
            self.status.set(
                "This frame failed to load. Select another "
                "frame or camera."
            )
            self.set_playing(False)
            return

        self.status.set(
            f"Loading {key[0]}, frame {key[1] + 1}... "
            "Previous frame remains clickable."
        )

        if self.future is None:
            self.pending_key = key
            self.future = self.executor.submit(
                self.load_frame, key
            )

    def set_index(self, index):
        self.index = max(
            0, min(len(self.infos) - 1, index)
        )
        self.timeline.set(self.index)
        self.request_frame()

    def set_playing(self, playing):
        self.playing = playing
        self.play_button.configure(
            text="Pause" if playing else "Play"
        )
        self.last_advance = time.monotonic()

    def toggle_play(self):
        if (
            not self.playing
            and self.index == len(self.infos) - 1
        ):
            self.set_index(0)

        self.set_playing(not self.playing)

    def step(self, amount):
        self.set_playing(False)
        self.set_index(self.index + amount)

    def seek_start(self, event):
        self.dragging = True
        self.set_playing(False)

    def seek_changed(self, value):
        if self.seek_job is not None:
            self.window.after_cancel(self.seek_job)

        self.seek_job = self.window.after(
            100, self.apply_seek
        )

    def apply_seek(self):
        self.seek_job = None
        self.set_index(int(round(self.timeline.get())))

    def seek_end(self, event):
        self.dragging = False

        if self.seek_job is not None:
            self.window.after_cancel(self.seek_job)
            self.seek_job = None

        self.apply_seek()

    def change_camera(self, event=None):
        # Preserve playback and the selected query.
        self.request_frame()

    def clear_query(self):
        self.query = None
        self.query_source = None
        self.query_patch = None
        self.query_preview = None
        self.score_key = None
        self.score_map = None

        self.query_text.set(
            "Click the left image to select a feature."
        )
        self.render()

    def click_image(self, event):
        if (
            self.features is None
            or self.image is None
            or self.display_geometry is None
        ):
            return

        (
            offset_x,
            offset_y,
            display_width,
            display_height,
            source_width,
            source_height,
        ) = self.display_geometry

        x = (
            (event.x - offset_x)
            * source_width / display_width
        )
        y = (
            (event.y - offset_y)
            * source_height / display_height
        )

        height, width = self.image.shape[:2]

        # Accept only the visible left image panel.
        if not (
            0 <= x < width
            and 0 <= y < height
        ):
            return

        _, grid_height, grid_width = self.features.shape

        patch_x = min(
            grid_width - 1,
            int(x * grid_width / width),
        )
        patch_y = min(
            grid_height - 1,
            int(y * grid_height / height),
        )

        # Query the frame actually displayed, even if the
        # next frame is currently being computed.
        self.query = self.features[
            :, patch_y, patch_x
        ].clone()
        self.query_source = self.loaded_key
        self.query_patch = patch_x, patch_y
        self.score_key = None
        self.score_map = None

        x0 = int(patch_x * width / grid_width)
        x1 = int((patch_x + 1) * width / grid_width)
        y0 = int(patch_y * height / grid_height)
        y1 = int((patch_y + 1) * height / grid_height)

        self.query_preview = self.image[
            y0:y1, x0:x1
        ].copy()

        camera_name, index = self.query_source

        self.query_text.set(
            f"Query: {camera_name}, frame {index + 1}, "
            f"patch ({patch_x}, {patch_y}). "
            "Selection stays active across frames and cameras."
        )

        print(
            f"Selected {camera_name}, frame {index + 1}, "
            f"patch ({patch_x}, {patch_y}); "
            f"playback={'playing' if self.playing else 'paused'}",
            flush=True,
        )

        # Deliberately do not pause playback.
        self.render()

    def render(self):
        if self.image is None or self.loaded_key is None:
            return

        camera_panel = self.image.copy()
        similarity_panel = np.zeros_like(camera_panel)
        height, width = camera_panel.shape[:2]

        lower = self.floor.get() / 100.0
        self.floor_label.configure(text=f"{lower:.2f}")

        if self.query is not None:
            if self.score_key != self.loaded_key:
                self.score_map = torch.einsum(
                    "c,chw->hw",
                    self.query,
                    self.features,
                ).numpy()
                self.score_key = self.loaded_key

            response = np.clip(
                (self.score_map - lower)
                / max(1e-6, 1.0 - lower),
                0.0,
                1.0,
            )

            response = cv2.resize(
                response,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            similarity_panel = cv2.applyColorMap(
                np.round(response * 255).astype(np.uint8),
                cv2.COLORMAP_VIRIDIS,
            )

            bar_width = min(220, width - 30)
            gradient = np.linspace(
                0, 255, bar_width
            ).astype(np.uint8)[None, :]

            color_bar = cv2.applyColorMap(
                np.repeat(gradient, 10, axis=0),
                cv2.COLORMAP_VIRIDIS,
            )

            similarity_panel[
                height - 20:height - 10,
                15:15 + bar_width,
            ] = color_bar

        else:
            similarity_panel[:] = (45, 20, 65)

            cv2.putText(
                similarity_panel,
                "Click the left image",
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        camera_name, index = self.loaded_key

        if self.boxes.get():
            draw_boxes(
                camera_panel,
                self.infos[index],
                camera_name,
                self.original_size,
            )

        if (
            self.query is not None
            and self.query_source == self.loaded_key
        ):
            patch_x, patch_y = self.query_patch
            _, grid_height, grid_width = self.features.shape

            p1 = (
                int(patch_x * width / grid_width),
                int(patch_y * height / grid_height),
            )
            p2 = (
                int((patch_x + 1) * width / grid_width) - 1,
                int((patch_y + 1) * height / grid_height) - 1,
            )

            for panel in (camera_panel, similarity_panel):
                cv2.rectangle(
                    panel,
                    p1,
                    p2,
                    (255, 255, 255),
                    2,
                )

        if self.query_preview is not None:
            preview_size = min(64, height // 4, width // 4)

            preview = cv2.resize(
                self.query_preview,
                (preview_size, preview_size),
                interpolation=cv2.INTER_NEAREST,
            )

            left = width - preview_size - 10
            top = 10

            similarity_panel[
                top:top + preview_size,
                left:left + preview_size,
            ] = preview

            cv2.rectangle(
                similarity_panel,
                (left, top),
                (left + preview_size, top + preview_size),
                (255, 255, 255),
                1,
            )

        self.last_display = np.hstack([
            camera_panel,
            similarity_panel,
        ])

        self.refresh_display()

    def schedule_resize(self, event=None):
        if self.resize_job is not None:
            self.window.after_cancel(self.resize_job)

        self.resize_job = self.window.after(
            40, self.refresh_display
        )

    def refresh_display(self):
        self.resize_job = None

        if self.last_display is None:
            return

        available_width = self.viewer.winfo_width()
        available_height = self.viewer.winfo_height()

        if available_width < 2 or available_height < 2:
            return

        source_height, source_width = (
            self.last_display.shape[:2]
        )

        scale = min(
            available_width / source_width,
            available_height / source_height,
        )

        display_width = max(
            1, int(source_width * scale)
        )
        display_height = max(
            1, int(source_height * scale)
        )

        offset_x = (
            available_width - display_width
        ) // 2
        offset_y = (
            available_height - display_height
        ) // 2

        rgb = cv2.cvtColor(
            self.last_display,
            cv2.COLOR_BGR2RGB,
        )

        display_image = Image.fromarray(rgb).resize(
            (display_width, display_height),
            Image.Resampling.BILINEAR,
        )

        self.photo = ImageTk.PhotoImage(display_image)

        self.viewer.coords(
            self.canvas_image,
            offset_x,
            offset_y,
        )
        self.viewer.itemconfigure(
            self.canvas_image,
            image=self.photo,
        )

        self.display_geometry = (
            offset_x,
            offset_y,
            display_width,
            display_height,
            source_width,
            source_height,
        )

    def save(self):
        if self.last_display is None:
            return

        output_directory = (
            self.root_path / "outputs" / "interactive"
        )
        output_directory.mkdir(
            parents=True, exist_ok=True
        )

        camera_name, index = self.loaded_key
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        output_path = output_directory / (
            f"{camera_name}_{index:04d}_{timestamp}.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            self.last_display,
        )

        if success:
            self.status.set(
                f"Saved {output_path.name}"
            )
        else:
            self.status.set("Screenshot saving failed.")

    def get_target_fps(self):
        try:
            return max(0.1, float(self.fps.get()))
        except (ValueError, tk.TclError):
            return max(0.1, self.args.fps)

    def tick(self):
        if self.closed:
            return

        if self.future is not None and self.future.done():
            finished_key = self.pending_key
            finished_future = self.future

            self.future = None
            self.pending_key = None

            try:
                result = finished_future.result()

            except Exception as error:
                self.failed_keys.add(finished_key)
                self.set_playing(False)

                self.status.set(
                    f"Frame loading failed: {error}"
                )

                messagebox.showerror(
                    "Frame loading failed",
                    str(error),
                    parent=self.window,
                )

            else:
                if finished_key == self.current_key():
                    (
                        self.image,
                        self.features,
                        self.original_size,
                    ) = result

                    self.loaded_key = finished_key
                    self.score_key = None
                    self.score_map = None

                    self.status.set(
                        f"Ready — {finished_key[0]}, "
                        f"frame {finished_key[1] + 1}"
                    )

                    self.last_advance = time.monotonic()
                    self.render()

                else:
                    # The user sought elsewhere while the
                    # worker was computing the old request.
                    self.request_frame()

        if (
            self.playing
            and not self.dragging
            and self.future is None
            and self.loaded_key == self.current_key()
            and time.monotonic() - self.last_advance
            >= 1.0 / self.get_target_fps()
        ):
            if self.index >= len(self.infos) - 1:
                self.set_playing(False)

            else:
                self.set_index(self.index + 1)
                self.last_advance = time.monotonic()

        self.tick_job = self.window.after(
            30, self.tick
        )

    def close(self):
        if self.closed:
            return

        self.closed = True
        self.playing = False

        for job in (
            self.seek_job,
            self.resize_job,
            self.tick_job,
        ):
            if job is not None:
                try:
                    self.window.after_cancel(job)
                except tk.TclError:
                    pass

        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        self.window.destroy()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Loading DINOv3 on {device}...",
        flush=True,
    )

    model = torch.hub.load(
        str(args.repo.resolve()),
        "dinov3_vits16",
        source="local",
        weights=str(args.weights.resolve()),
    ).to(device).eval()

    model.requires_grad_(False)

    # Only load pickle files from a trusted source.
    with args.ann.open("rb") as handle:
        dataset = pickle.load(handle)

    infos = sorted(
        dataset["infos"],
        key=lambda item: item["timestamp"],
    )

    if not infos:
        raise ValueError("No annotation records found.")

    print(
        f"Loaded {len(infos)} annotation frames. "
        "Opening fullscreen player...",
        flush=True,
    )

    window = tk.Tk()
    player = DinoPlayer(
        window,
        args,
        model,
        infos,
        device,
    )
    window.mainloop()


if __name__ == "__main__":
    main()
