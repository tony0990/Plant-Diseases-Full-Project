import os
import sys
import time
import random
import numpy as np
from pathlib import Path

import pygame

# Optional: for loading Keras models & image preprocessing
import tensorflow as tf
from PIL import Image

# =========================
# PATHS (عدلهم لو لزم)
# =========================
PROJECT_ROOT = Path(r"C:\Users\tonyh\Documents\GitHub\Plant-Diseases-Full-Project")
FX_DIR = PROJECT_ROOT / "fx"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DATASET_DIR = PROJECT_ROOT / "test dataset"

INTRO_IMG = FX_DIR / "intro.png"
MODEL_SCREEN_IMG = FX_DIR / "model1.png"
MUSIC_FILE = FX_DIR / "music.wav"

# Keras model files (حسب الصورة اللي عندك)
MODEL_FILES = {
    "EfficientNetB3": MODELS_DIR / "best_finetuned_model.keras",
    "ResNet50": MODELS_DIR / "resnet50_finetuned_final.keras",
    "InceptionV3": MODELS_DIR / "inception_model.keras",
    "DenseNet121": MODELS_DIR / "densenet_best_model.keras",
    "MobileNetV2": None,  # under development
}

# input size options
IMG_SIZE_OPTIONS = [(224, 224), (256, 256)]
DEFAULT_IMG_SIZE = (224, 224)

# =========================
# UI helpers
# =========================
WHITE = (255, 255, 255)
BLACK = (10, 10, 10)
DARK = (25, 25, 25)
GREEN = (60, 200, 120)
RED = (220, 80, 80)
BLUE = (80, 150, 220)
GRAY = (180, 180, 180)

pygame.init()
pygame.mixer.init()

FONT = pygame.font.SysFont("arial", 22)
FONT_BIG = pygame.font.SysFont("arial", 34, bold=True)
FONT_SMALL = pygame.font.SysFont("arial", 18)


def load_image_scaled_exact(path: Path):
    img = pygame.image.load(str(path)).convert_alpha()
    w, h = img.get_width(), img.get_height()
    return img, (w, h)


class Button:
    def __init__(self, rect, text, bg=(40,40,40), fg=WHITE):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.bg = bg
        self.fg = fg
        self.hover = False

    def draw(self, screen):
        color = tuple(min(255, c + 30) for c in self.bg) if self.hover else self.bg
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, (255,255,255), self.rect, 2, border_radius=10)

        txt = FONT.render(self.text, True, self.fg)
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


def draw_loading_bar(screen, x, y, w, h, progress_0_1):
    pygame.draw.rect(screen, (255,255,255), (x, y, w, h), 2, border_radius=8)
    fill_w = int((w - 4) * max(0.0, min(1.0, progress_0_1)))
    pygame.draw.rect(screen, GREEN, (x + 2, y + 2, fill_w, h - 4), border_radius=8)


def list_test_images(test_root: Path):
    # expects structure: test dataset / class_name / images...
    class_dirs = [d for d in test_root.iterdir() if d.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())
    images = []
    for cls_dir in class_dirs:
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                images.append((cls_dir.name, p))
    return images


def load_labels(test_root: Path):
    labels_path = test_root / "labels.txt"
    if labels_path.exists():
        return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # fallback: folder names
    return sorted([d.name for d in test_root.iterdir() if d.is_dir()])


def preprocess_for_model(img_path: Path, img_size):
    # We assume the preprocessing layers are inside the saved model (as in your setup),
    # so we only resize and cast to float32.
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img, dtype=np.float32)  # (H,W,3)
    arr = np.expand_dims(arr, axis=0)      # (1,H,W,3)
    return arr, np.array(img)              # return both model input + display image


def predict_with_model(keras_model, img_input, class_names):
    preds = keras_model.predict(img_input, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    return label, conf, preds[0]


def safe_load_keras_model(path: Path):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return tf.keras.models.load_model(str(path))


def main():
    # ---- Load intro image and set window size to image size
    intro_surface, (W, H) = load_image_scaled_exact(INTRO_IMG)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Plant Diseases - Full Project")

    # ---- Background music
    if MUSIC_FILE.exists():
        pygame.mixer.music.load(str(MUSIC_FILE))
        pygame.mixer.music.play(-1)  # loop forever

    clock = pygame.time.Clock()

    # =========================
    # INTRO SCREEN + LOADING
    # =========================
    start_time = time.time()
    loading_duration = 2.5  # seconds (feel free)
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / loading_duration)

        screen.blit(intro_surface, (0, 0))

        # loading bar at bottom
        bar_w = int(W * 0.7)
        bar_h = 18
        bar_x = (W - bar_w) // 2
        bar_y = H - 40
        draw_loading_bar(screen, bar_x, bar_y, bar_w, bar_h, progress)

        loading_txt = FONT_SMALL.render("Loading...", True, WHITE)
        screen.blit(loading_txt, (bar_x, bar_y - 22))

        pygame.display.flip()

        if progress >= 1.0:
            break

    # =========================
    # MODEL SELECTION SCREEN
    # =========================
    model_surface, (W2, H2) = load_image_scaled_exact(MODEL_SCREEN_IMG)

    # Reset window size to model1.png size
    screen = pygame.display.set_mode((W2, H2))

    # Buttons layout (adjust positions if needed)
    btn_w = int(W2 * 0.42)
    btn_h = 52
    left = int(W2 * 0.08)
    top = int(H2 * 0.22)
    gap = 18

    model_names = ["EfficientNetB3", "ResNet50", "InceptionV3", "MobileNetV2", "DenseNet121"]
    buttons = []
    for i, name in enumerate(model_names):
        y = top + i * (btn_h + gap)
        bg = (50, 50, 50) if name != "MobileNetV2" else (70, 60, 60)
        buttons.append(Button((left, y, btn_w, btn_h), name, bg=bg))

    # image size toggle buttons
    toggle_224 = Button((left, top + 5*(btn_h+gap) + 10, int(btn_w*0.48), 45), "Input 224x224", bg=(35,55,85))
    toggle_256 = Button((left + int(btn_w*0.52), top + 5*(btn_h+gap) + 10, int(btn_w*0.48), 45), "Input 256x256", bg=(35,55,85))
    current_img_size = DEFAULT_IMG_SIZE

    # status text
    status_msg = "Choose a model, then the app will test on a random image from test dataset."
    status_color = WHITE

    # load class names from labels.txt
    class_names = load_labels(TEST_DATASET_DIR)

    # cached loaded models to avoid reloading each click
    loaded_models = {}

    # main loop
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            # toggle input size
            if toggle_224.handle_event(event):
                current_img_size = (224, 224)
                status_msg = "Input size set to 224x224"
                status_color = GREEN
            if toggle_256.handle_event(event):
                current_img_size = (256, 256)
                status_msg = "Input size set to 256x256"
                status_color = GREEN

            # model buttons
            for btn in buttons:
                if btn.handle_event(event):
                    chosen = btn.text

                    if chosen == "MobileNetV2":
                        status_msg = "MobileNetV2 is under development (GUI ready)."
                        status_color = RED
                        break

                    model_path = MODEL_FILES.get(chosen)
                    try:
                        if chosen not in loaded_models:
                            status_msg = f"Loading {chosen} model..."
                            status_color = BLUE
                            # quick UI update
                            screen.blit(model_surface, (0, 0))
                            for b in buttons:
                                b.draw(screen)
                            toggle_224.draw(screen)
                            toggle_256.draw(screen)
                            pygame.display.flip()

                            loaded_models[chosen] = safe_load_keras_model(model_path)

                        keras_model = loaded_models[chosen]

                        # pick a random test image
                        all_test_images = list_test_images(TEST_DATASET_DIR)
                        if not all_test_images:
                            status_msg = "No images found in test dataset folder. Run prepare_test_dataset.py first."
                            status_color = RED
                            break

                        true_class, img_path = random.choice(all_test_images)

                        img_input, img_display = preprocess_for_model(img_path, current_img_size)
                        pred_label, conf, probs = predict_with_model(keras_model, img_input, class_names)

                        status_msg = f"[{chosen}] True: {true_class} | Pred: {pred_label} | Conf: {conf:.2%}"
                        status_color = GREEN if pred_label == true_class else RED

                        # show the chosen image on the right side
                        # draw it scaled to fit a box
                        box_x = int(W2 * 0.55)
                        box_y = int(H2 * 0.22)
                        box_w = int(W2 * 0.40)
                        box_h = int(H2 * 0.55)

                        # convert PIL/np to pygame surface
                        surf = pygame.surfarray.make_surface(np.transpose(img_display, (1, 0, 2)))
                        # scale to fit
                        surf = pygame.transform.smoothscale(surf, (box_w, box_h))

                        # store for drawing
                        last_preview = (surf, (box_x, box_y), str(img_path))

                        # also store last probs for simple display
                        last_probs = (chosen, pred_label, conf, probs)

                    except Exception as e:
                        status_msg = f"Error: {e}"
                        status_color = RED
                    break

        # draw background
        screen.blit(model_surface, (0, 0))

        # draw buttons
        for btn in buttons:
            btn.draw(screen)

        toggle_224.draw(screen)
        toggle_256.draw(screen)

        # draw status box
        pygame.draw.rect(screen, (0, 0, 0), (0, H2 - 60, W2, 60))
        txt = FONT_SMALL.render(status_msg, True, status_color)
        screen.blit(txt, (16, H2 - 42))

        # draw preview + probs if available
        if "last_preview" in locals():
            surf, pos, imgpath = last_preview
            screen.blit(surf, pos)

            # image path
            path_txt = FONT_SMALL.render(f"Test image: {Path(imgpath).name}", True, WHITE)
            screen.blit(path_txt, (pos[0], pos[1] + surf.get_height() + 8))

        if "last_probs" in locals():
            chosen, pred_label, conf, probs = last_probs
            # show top-3
            top3 = np.argsort(probs)[::-1][:3]
            x0 = int(W2 * 0.55)
            y0 = int(H2 * 0.82)
            title = FONT.render("Top-3 predictions:", True, WHITE)
            screen.blit(title, (x0, y0))

            for i, idx in enumerate(top3):
                name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                line = f"{i+1}) {name}: {probs[idx]*100:.2f}%"
                t = FONT_SMALL.render(line, True, GRAY)
                screen.blit(t, (x0, y0 + 28 + i*22))

        pygame.display.flip()


if __name__ == "__main__":
    # Quick sanity checks
    missing = []
    for p in [INTRO_IMG, MODEL_SCREEN_IMG, MUSIC_FILE]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("Missing assets:")
        for m in missing:
            print(" -", m)
        print("\nFix paths / files then re-run.")
        sys.exit(1)

    if not TEST_DATASET_DIR.exists():
        print(f"Test dataset not found: {TEST_DATASET_DIR}")
        print("Run prepare_test_dataset.py first.")
        sys.exit(1)

    main()
