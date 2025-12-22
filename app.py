"""
app.py
Plant Disease Detection Application
Supports multiple CNN models with proper preprocessing for each
"""

import sys
import time
from pathlib import Path
import ctypes

import pygame
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================================
# WINDOW CONFIG
# ==========================================================
APP_WIDTH = 900
APP_HEIGHT = 550

# ==========================================================
# PATHS
# ==========================================================
PROJECT_ROOT = Path(r"C:\Users\tonyh\Documents\GitHub\Plant-Diseases-Full-Project")
FX_DIR = PROJECT_ROOT / "fx"
ALL_MODELS_DIR = PROJECT_ROOT / "All Models"
TEST_DATASET_DIR = PROJECT_ROOT / "test dataset"

INTRO_IMG = FX_DIR / "intro.png"
MODEL_BG_IMG = FX_DIR / "model1.png"
MUSIC_FILE = FX_DIR / "music.wav"

# Model files - EXACT NAMES FROM YOUR SCREENSHOT
MODEL_FILES = {
    "EfficientNetB3": ALL_MODELS_DIR / "best_finetuned_model.keras",
    "ResNet50": ALL_MODELS_DIR / "resnet50_finetuned_final.keras",
    "InceptionV3": ALL_MODELS_DIR / "inception_model.keras",
    "DenseNet121": ALL_MODELS_DIR / "densenet_best_model.keras",
}

# Model input sizes and preprocessing functions
MODEL_CONFIG = {
    "EfficientNetB3": {
        "size": (256, 256),
        "preprocess": tf.keras.applications.efficientnet.preprocess_input
    },
    "ResNet50": {
        "size": (224, 224),
        "preprocess": tf.keras.applications.resnet50.preprocess_input
    },
    "InceptionV3": {
        "size": (299, 299),  # InceptionV3 uses 299x299
        "preprocess": tf.keras.applications.inception_v3.preprocess_input
    },
    "DenseNet121": {
        "size": (256, 256),  # Using 256x256 as requested
        "preprocess": tf.keras.applications.densenet.preprocess_input
    },
}

# ==========================================================
# INIT
# ==========================================================
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((APP_WIDTH, APP_HEIGHT))
pygame.display.set_caption("Plant Disease Detection - Multi-Model")

FONT = pygame.font.SysFont("arial", 22)
FONT_SMALL = pygame.font.SysFont("arial", 18)
FONT_TINY = pygame.font.SysFont("arial", 14)

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (80, 80, 80)
GREEN = (60, 200, 120)
RED = (220, 80, 80)
BLUE = (80, 150, 220)
YELLOW = (220, 180, 60)

# ==========================================================
# WINDOWS FILE DIALOG
# ==========================================================
def open_file_dialog():
    """Opens Windows file dialog to select an image file."""
    buffer = ctypes.create_unicode_buffer(1024)

    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", ctypes.c_uint32),
            ("hwndOwner", ctypes.c_void_p),
            ("hInstance", ctypes.c_void_p),
            ("lpstrFilter", ctypes.c_wchar_p),
            ("lpstrCustomFilter", ctypes.c_wchar_p),
            ("nMaxCustFilter", ctypes.c_uint32),
            ("nFilterIndex", ctypes.c_uint32),
            ("lpstrFile", ctypes.c_wchar_p),
            ("nMaxFile", ctypes.c_uint32),
            ("lpstrFileTitle", ctypes.c_wchar_p),
            ("nMaxFileTitle", ctypes.c_uint32),
            ("lpstrInitialDir", ctypes.c_wchar_p),
            ("lpstrTitle", ctypes.c_wchar_p),
            ("Flags", ctypes.c_uint32),
            ("nFileOffset", ctypes.c_uint16),
            ("nFileExtension", ctypes.c_uint16),
            ("lpstrDefExt", ctypes.c_wchar_p),
            ("lCustData", ctypes.c_void_p),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", ctypes.c_wchar_p),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", ctypes.c_uint32),
            ("FlagsEx", ctypes.c_uint32),
        ]

    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    ofn.hwndOwner = None
    ofn.lpstrFilter = "Images\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0"
    ofn.lpstrFile = ctypes.cast(buffer, ctypes.c_wchar_p)
    ofn.nMaxFile = 1024
    ofn.lpstrTitle = "Select Plant Image"
    ofn.Flags = 0x00080000 | 0x00001000  # OFN_EXPLORER | OFN_FILEMUSTEXIST

    try:
        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return Path(buffer.value)
    except Exception as e:
        print(f"File dialog error: {e}")

    return None

# ==========================================================
# HELPERS
# ==========================================================
def load_labels():
    """Load class labels from labels.txt file."""
    labels_file = TEST_DATASET_DIR / "labels.txt"
    if not labels_file.exists():
        print(f"Warning: {labels_file} not found!")
        return []

    labels = labels_file.read_text(encoding='utf-8').splitlines()
    print(f"Loaded {len(labels)} class labels")
    return labels


def preprocess_image(path, model_name):
    """
    Preprocess image for model prediction using correct preprocessing.
    Each model has its own preprocessing function and input size.
    """
    try:
        # Load image
        img = Image.open(path).convert("RGB")
        display = img.copy()

        # Get model-specific config
        config = MODEL_CONFIG[model_name]
        target_size = config["size"]
        preprocess_fn = config["preprocess"]

        # Resize for model input
        img_resized = img.resize(target_size)

        # Convert to array and add batch dimension
        arr = np.array(img_resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        # Apply model-specific preprocessing
        arr = preprocess_fn(arr)

        return arr, display

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None


def draw_text_with_shadow(surface, text, font, color, pos, shadow_color=BLACK):
    """Draw text with shadow for better visibility."""
    # Shadow
    shadow_surf = font.render(text, True, shadow_color)
    surface.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
    # Main text
    text_surf = font.render(text, True, color)
    surface.blit(text_surf, pos)


# ==========================================================
# MAIN
# ==========================================================
def main():
    clock = pygame.time.Clock()

    # Load images
    try:
        intro = pygame.transform.smoothscale(
            pygame.image.load(str(INTRO_IMG)), (APP_WIDTH, APP_HEIGHT)
        )
        bg = pygame.transform.smoothscale(
            pygame.image.load(str(MODEL_BG_IMG)), (APP_WIDTH, APP_HEIGHT)
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        # Create fallback backgrounds
        intro = pygame.Surface((APP_WIDTH, APP_HEIGHT))
        intro.fill((20, 40, 60))
        bg = pygame.Surface((APP_WIDTH, APP_HEIGHT))
        bg.fill((30, 30, 30))

    # Load music (optional)
    try:
        if MUSIC_FILE.exists():
            pygame.mixer.music.load(str(MUSIC_FILE))
            pygame.mixer.music.set_volume(0.3)
            pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Music loading skipped: {e}")

    # Intro screen with 4-second loading bar
    start = time.time()
    loading_duration = 4.0

    while time.time() - start < loading_duration:
        screen.blit(intro, (0, 0))
        progress = (time.time() - start) / loading_duration

        # Draw loading bar
        bar_x, bar_y = 150, 480
        bar_width, bar_height = 600, 14
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        pygame.draw.rect(screen, GREEN,
                        (bar_x + 2, bar_y + 2, int((bar_width - 4) * progress), bar_height - 4))

        # Loading text
        loading_text = FONT.render("Loading Plant Disease Detection...", True, WHITE)
        screen.blit(loading_text, (APP_WIDTH // 2 - loading_text.get_width() // 2, 450))

        pygame.display.flip()
        clock.tick(60)

        # Handle events during loading
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    # Load class labels
    class_names = load_labels()
    if not class_names:
        print("⚠ Warning: No class labels loaded! Using generic labels.")
        class_names = [f"Class {i}" for i in range(5)]

    # Model storage
    models = {}
    model_names = list(MODEL_FILES.keys())

    # Application state
    state = "select"  # "select" or "predict"
    selected_model = None
    preview = None
    status = "Choose a model to begin"
    prediction_text = ""

    # Main loop
    running = True
    while running:
        clock.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                x, y = e.pos

                if state == "select":
                    # Check model selection buttons
                    for i, name in enumerate(model_names):
                        rect = pygame.Rect(50, 120 + i * 55, 360, 45)
                        if rect.collidepoint(x, y):
                            selected_model = name
                            status = f"Loading {name}..."
                            pygame.display.flip()

                            # Load model if not already loaded
                            if name not in models:
                                try:
                                    model_path = MODEL_FILES[name]
                                    if not model_path.exists():
                                        status = f"❌ Model file not found: {model_path.name}"
                                        print(f"Error: {model_path} does not exist!")
                                        continue

                                    print(f"Loading model: {name} from {model_path}")
                                    models[name] = tf.keras.models.load_model(
                                        str(model_path), compile=False
                                    )
                                    status = f"✓ {name} loaded. Upload an image."
                                    print(f"Model {name} loaded successfully!")

                                except Exception as e:
                                    status = f"❌ Error loading {name}"
                                    print(f"Error loading model {name}: {e}")
                                    continue
                            else:
                                status = f"✓ {name} ready. Upload an image."

                            state = "predict"
                            prediction_text = ""

                elif state == "predict":
                    # Buttons raised higher
                    upload_rect = pygame.Rect(50, 360, 360, 45)
                    back_rect = pygame.Rect(50, 420, 360, 45)

                    if upload_rect.collidepoint(x, y):
                        path = open_file_dialog()
                        if path:
                            status = "Processing image..."
                            prediction_text = ""
                            pygame.display.flip()

                            inp, disp = preprocess_image(path, selected_model)
                            if inp is not None and disp is not None:
                                try:
                                    # Predict
                                    preds = models[selected_model].predict(inp, verbose=0)[0]
                                    idx = int(np.argmax(preds))
                                    confidence = preds[idx] * 100

                                    # Get class name
                                    if idx < len(class_names):
                                        class_name = class_names[idx]
                                    else:
                                        class_name = f"Class {idx}"

                                    # Update status
                                    status = f"✓ Prediction complete"
                                    prediction_text = f"{class_name}\n{confidence:.2f}% confidence"
                                    print(f"Prediction: {class_name} ({confidence:.2f}%)")

                                    # Create preview
                                    surf = pygame.surfarray.make_surface(
                                        np.transpose(np.array(disp), (1, 0, 2))
                                    )
                                    preview = pygame.transform.smoothscale(surf, (420, 300))

                                except Exception as e:
                                    status = f"❌ Prediction error"
                                    prediction_text = ""
                                    print(f"Prediction error: {e}")
                            else:
                                status = "❌ Error loading image"
                                prediction_text = ""

                    if back_rect.collidepoint(x, y):
                        state = "select"
                        preview = None
                        status = "Choose a model to begin"
                        prediction_text = ""

        # Draw background
        screen.blit(bg, (0, 0))

        if state == "select":
            # Draw title
            title = FONT.render("Select a Model", True, WHITE)
            screen.blit(title, (50, 70))

            # Draw model selection buttons
            for i, name in enumerate(model_names):
                rect = pygame.Rect(50, 120 + i * 55, 360, 45)

                # Check if model is loaded
                is_loaded = name in models
                btn_color = GREEN if is_loaded else GRAY

                pygame.draw.rect(screen, btn_color, rect, border_radius=8)
                pygame.draw.rect(screen, WHITE, rect, 2, border_radius=8)

                text = FONT.render(name, True, WHITE)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

                # Show checkmark if loaded
                if is_loaded:
                    check = FONT_SMALL.render("✓", True, WHITE)
                    screen.blit(check, (rect.right - 30, rect.centery - 10))

        else:  # state == "predict"
            # Draw title with model name
            title = FONT.render(f"Model: {selected_model}", True, WHITE)
            screen.blit(title, (50, 70))

            # Draw action buttons - raised positions
            upload_rect = pygame.Rect(50, 360, 360, 45)
            back_rect = pygame.Rect(50, 420, 360, 45)

            # Upload button
            pygame.draw.rect(screen, BLUE, upload_rect, border_radius=8)
            pygame.draw.rect(screen, WHITE, upload_rect, 2, border_radius=8)
            upload_text = FONT.render("📁 Upload Image", True, WHITE)
            upload_text_rect = upload_text.get_rect(center=upload_rect.center)
            screen.blit(upload_text, upload_text_rect)

            # Back button
            pygame.draw.rect(screen, RED, back_rect, border_radius=8)
            pygame.draw.rect(screen, WHITE, back_rect, 2, border_radius=8)
            back_text = FONT.render("← Back", True, WHITE)
            back_text_rect = back_text.get_rect(center=back_rect.center)
            screen.blit(back_text, back_text_rect)

            # Draw preview and prediction
            if preview:
                # Draw preview with border
                preview_rect = pygame.Rect(450, 120, 420, 300)
                pygame.draw.rect(screen, WHITE, preview_rect, 2)
                screen.blit(preview, (450, 120))

                # Draw prediction text below preview
                if prediction_text:
                    lines = prediction_text.split('\n')
                    y_offset = 440

                    # Draw semi-transparent background for text
                    text_bg = pygame.Surface((400, 60))
                    text_bg.fill(BLACK)
                    text_bg.set_alpha(180)
                    screen.blit(text_bg, (455, y_offset - 10))

                    # Draw prediction text
                    for line in lines:
                        if "%" in line:
                            text_surf = FONT.render(line, True, YELLOW)
                        else:
                            text_surf = FONT.render(line, True, GREEN)
                        screen.blit(text_surf, (460, y_offset))
                        y_offset += 25

        # Draw status bar at bottom
        pygame.draw.rect(screen, BLACK, (0, 500, APP_WIDTH, 50))
        status_text = FONT_SMALL.render(status, True, WHITE)
        screen.blit(status_text, (20, 515))

        # Draw model info in corner
        if selected_model and state == "predict":
            config = MODEL_CONFIG[selected_model]
            info_text = FONT_TINY.render(
                f"Input: {config['size'][0]}x{config['size'][1]}",
                True, GRAY
            )
            screen.blit(info_text, (APP_WIDTH - 120, 515))

        pygame.display.flip()

    # Cleanup
    pygame.quit()


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("="*70)
    print("Plant Disease Detection - Multi-Model Application")
    print("="*70)
    print("\nSupported Models:")
    for name in MODEL_FILES.keys():
        print(f"  - {name}")
    print("\nStarting application...")
    print("="*70)

    main()