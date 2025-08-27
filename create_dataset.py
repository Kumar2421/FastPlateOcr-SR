import os, random, csv, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import matplotlib.font_manager as fm

# -----------------------------
# Font Handling (system fonts only)
# -----------------------------
system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
FONT_PATHS = [f for f in system_fonts if any(name in f.lower() for name in [
    "dejavu", "liberation", "arial", "noto sans", "roboto"
])]

def get_font(size=52):
    """Try to load a system font, else fallback."""
    if FONT_PATHS:
        try:
            return ImageFont.truetype(random.choice(FONT_PATHS), size)
        except OSError:
            pass
    return ImageFont.load_default()

# -----------------------------
# Output config
# -----------------------------
OUT_DIR = "synthetic_plates2"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_FILE = os.path.join(OUT_DIR, "plates.csv")

# -----------------------------
# Augmentations (fixed params)
# -----------------------------
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=50, p=0.3),   # compatible
    A.Perspective(scale=(0.01, 0.03), keep_size=True, p=0.3),
    A.Affine(rotate=(-2, 2), shear=(-2, 2), p=0.4),
    A.ImageCompression(quality_range=(50, 90), p=0.3),
    A.Downscale(scale=(0.7, 0.9), p=0.2),   # compatible
    A.RandomShadow(shadow_dimension=5, p=0.2),  # compatible
    A.RandomRain(brightness_coefficient=0.95, drop_length=5, drop_width=1, blur_value=2, p=0.1),
    A.CoarseDropout(max_holes=2, hole_height_range=(8, 15), hole_width_range=(15, 30), p=0.2),  # compatible
    A.ElasticTransform(alpha=10, sigma=5, p=0.1),  # compatible
    A.Sharpen(alpha=(0.1,0.3), lightness=(0.7,1.3), p=0.2),
])


# -----------------------------
# Backgrounds + States
# -----------------------------
bg_colors = [(255,255,255), (250,250,210), (255,255,180), (240,240,240)]

STATE_CODES = [
    "TN","KA","MH","DL","GJ","RJ","UP","WB","KL","AP",
    "TS","MP","HR","PB","CH","BR","JH","UK","HP","GA",
    "OD","AS","TR","MN","ML","SK","AR","NL","AN","LD",
    "PY","DN","DD","JK","LA"
]

# -----------------------------
# Helpers
# -----------------------------
def draw_centered_bold_text(draw, img_size, text, font, fill, y_offset=0):
    """Draw text in center with fake bold (outline)."""
    w, h = img_size
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x = (w - text_w) // 2
    y = ((h - text_h) // 2) + y_offset
    for dx, dy in [(0,0), (1,0), (0,1), (1,1)]:  # fake bold
        draw.text((x+dx, y+dy), text, font=font, fill=fill)

def make_plate(text, w=512, h=96, two_line=False):
    """Generate a synthetic plate with text + augmentations."""
    # --- Background ---
    base_color = random.choice(bg_colors)
    bg = np.ones((h, w, 3), dtype=np.uint8) * np.array(base_color, dtype=np.uint8)

    # Add slight gradient safely
    for i in range(h):
        shade = random.randint(-5, 5)
        row = bg[i].astype(np.int16) + shade
        bg[i] = np.clip(row, 0, 255).astype(np.uint8)

    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    # --- Border ---
    border_thickness = random.randint(2, 3)
    for t in range(border_thickness):
        draw.rectangle([t, t, w-t-1, h-t-1], outline=(0,0,0))

    # --- Text ---
    text_color = (0,0,0)
    if two_line:
        mid = len(text)//2
        parts = [text[:mid], text[mid:]]
        font1 = get_font(random.randint(42, 52))
        font2 = get_font(random.randint(42, 52))
        draw_centered_bold_text(draw, (w, h), parts[0], font1, text_color, y_offset=-20)
        draw_centered_bold_text(draw, (w, h), parts[1], font2, text_color, y_offset=20)
    else:
        font = get_font(random.randint(48, 58))
        draw_centered_bold_text(draw, (w, h), text, font, text_color)

    img = np.array(img)
    img = augment(image=img)["image"]
    return img

def rand_plate_text():
    """Generate random Indian-style plate text."""
    st = random.choice(STATE_CODES)
    district = f"{random.randint(1,99):02d}"
    series = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=random.choice([1,2])))
    number = f"{random.randint(0,9999):04d}"
    return f"{st}{district}{series}{number}"

# -----------------------------
# Generate Dataset
# -----------------------------
NUM_IMAGES = 6000

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","plate_text"])
    
    for i in range(1, NUM_IMAGES+1):
        text = rand_plate_text()
        two_line = random.random() < 0.3
        img = make_plate(text, two_line=two_line)
        filename = f"plate_{i:05d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.writerow([filename, text])
        if i % 6000 == 0:
            print(f"✅ Generated {i}/{NUM_IMAGES} plates")
