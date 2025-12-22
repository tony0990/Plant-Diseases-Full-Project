import os
import sys
import time
import random
from pathlib import Path

import pygame
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================================
# PATH CONFIGURATION (EDIT ONLY IF YOUR FOLDERS MOVE)
# ==========================================================
PROJECT_ROOT = Path(r"C:\Users\tonyh\Documents\GitHub\Plant-Diseases-Full-Project")
FX_DIR = PROJECT_ROOT / "fx"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DATASET_DIR = PROJECT_ROOT / "test dataset"

INTRO_IMG = FX_DIR / "intro.png"
MODEL_SCREEN_IMG = FX_DIR / "model1.png"
MUSIC_FILE = FX_DIR / "music.wav"

MODEL_FILES = {
    "EfficientNetB3": MODELS_DIR / "best_finetuned_model.keras",
    "ResNet50": MODELS_DIR / "resnet50_finetuned_final.keras",
    "InceptionV3": MODELS_DIR / "inception_model.keras",
    "DenseNet121": MODELS_DIR / "densenet_best_model.keras",
    "MobileNetV2": None  # under development
}

IMG_SIZE_224 = (224, 224)
IMG_SIZE_256 = (256, 256)
CURRENT_IMG_SIZE = IMG_SIZE_224

# ==========================================================
# PYGAME INIT (IMPORTANT ORDER)
# ==========================================================
pygame.init()
pygame.mixer.init()

FONT = pygame.font.SysFont("arial", 22)
FONT_SMALL = pygame.font.SysFont("arial", 18)
FONT_BIG = pygame.font.SysFont("arial", 34, bold=True)

WHITE = (255, 255, 255)
BLACK = (10, 10, 10)
GRAY = (180, 180, 180)
GREEN = (60, 200, 120)
RED = (220, 80, 80)
BLUE = (80, 150, 220)

# ==========================================================
# UI HELPERS
# ==========================================================
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.hover = False

    def draw(self, screen):
        color = (70, 70, 70) if not self.hover else (100, 100, 100)
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=10)
        txt = FONT.render(self.text, True, WHITE)
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


def draw_loading_bar(screen, x, y, w, h, progress):
    pygame.draw.rect(screen, WHITE, (x, y, w, h), 2, border_radius=8)
    fill = int((w - 4) * progress)
    pygame.draw.rect(screen, GREEN, (x + 2, y + 2, fill, h - 4), border_radius=8)


# ==========================================================
# DATA HELPERS
# ==========================================================
def load_labels():
    labels_file = TEST_DATASET_DIR / "labels.txt"
    return labels_file.read_text().splitlines()


def list_test_images():
    images = []
    for cls_dir in TEST_DATASET_DIR.iterdir():
        if cls_dir.is_dir():
            for img in cls_dir.glob("*"):
                if img.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    images.append((cls_dir.name, img))
    return images


def preprocess_image(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr, img


# ==========================================================
# MAIN APP
# ==========================================================
def main():
    # ------------------------------------------------------
    # INTRO SCREEN (FIXED ORDER)
    # ------------------------------------------------------
    intro_raw = pygame.image.load(str(INTRO_IMG))
    W, H = intro_raw.get_width(), intro_raw.get_height()

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Plant Diseases - Full Project")

    intro_surface = intro_raw.convert_alpha()

    pygame.mixer.music.load(str(MUSIC_FILE))
    pygame.mixer.music.play(-1)

    clock = pygame.time.Clock()
    start_time = time.time()

    while True:
        clock.tick(60)
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / 2.5)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(intro_surface, (0, 0))
        draw_loading_bar(screen, W * 0.15, H - 40, W * 0.7, 18, progress)
        pygame.display.flip()

        if progress >= 1:
            break

    # ------------------------------------------------------
    # MODEL SELECTION SCREEN
    # ------------------------------------------------------
    model_raw = pygame.image.load(str(MODEL_SCREEN_IMG))
    W2, H2 = model_raw.get_width(), model_raw.get_height()
    screen = pygame.display.set_mode((W2, H2))
    model_surface = model_raw.convert_alpha()

    buttons = []
    names = ["EfficientNetB3", "ResNet50", "InceptionV3", "MobileNetV2", "DenseNet121"]
    for i, n in enumerate(names):
        buttons.append(Button((60, 160 + i * 70, 380, 55), n))

    btn_224 = Button((60, 520, 180, 45), "224 x 224")
    btn_256 = Button((260, 520, 180, 45), "256 x 256")

    class_names = load_labels()
    test_images = list_test_images()
    loaded_models = {}

    status = "Select a model to test a random image from the test dataset."
    status_color = WHITE

    while True:
        clock.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if btn_224.handle(e):
                global CURRENT_IMG_SIZE
                CURRENT_IMG_SIZE = IMG_SIZE_224
                status = "Input size set to 224 x 224"
                status_color = GREEN

            if btn_256.handle(e):
                CURRENT_IMG_SIZE = IMG_SIZE_256
                status = "Input size set to 256 x 256"
                status_color = GREEN

            for b in buttons:
                if b.handle(e):
                    model_name = b.text

                    if model_name == "MobileNetV2":
                        status = "MobileNetV2 is under development."
                        status_color = RED
                        break

                    if model_name not in loaded_models:
                        loaded_models[model_name] = tf.keras.models.load_model(
                            str(MODEL_FILES[model_name])
                        )

                    true_label, img_path = random.choice(test_images)
                    inp, disp = preprocess_image(img_path, CURRENT_IMG_SIZE)

                    preds = loaded_models[model_name].predict(inp, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    conf = preds[idx]

                    pred_label = class_names[idx]
                    status = f"{model_name} | True: {true_label} | Pred: {pred_label} | Conf: {conf:.2%}"
                    status_color = GREEN if pred_label == true_label else RED

                    preview = pygame.surfarray.make_surface(
                        np.transpose(np.array(disp), (1, 0, 2))
                    )
                    preview = pygame.transform.smoothscale(preview, (420, 320))

        screen.blit(model_surface, (0, 0))
        for b in buttons:
            b.draw(screen)

        btn_224.draw(screen)
        btn_256.draw(screen)

        pygame.draw.rect(screen, BLACK, (0, H2 - 60, W2, 60))
        screen.blit(FONT_SMALL.render(status, True, status_color), (20, H2 - 42))

        if 'preview' in locals():
            screen.blit(preview, (520, 180))

        pygame.display.flip()


if __name__ == "__main__":
    main()
