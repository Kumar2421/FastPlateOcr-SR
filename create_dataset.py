import os, random, csv, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

# Fonts (update paths if needed)
FONT_PATHS = [
    "arialbd.ttf", "bahnschrift.ttf", "consola.ttf"
]

OUT_DIR = "synthetic_plates"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUT_DIR, "plates.csv")

# ✅ Mid-level realistic augmentations
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(5, 10), p=0.3),    # lower noise, more natural
    A.Perspective(scale=(0.01,0.03), keep_size=True, p=0.3),
    A.Affine(rotate=(-2, 2), shear=(-2, 2), p=0.4),  # slight tilt
    A.ImageCompression(quality_range=(50,90), p=0.3),  # realistic JPEG
    A.Downscale(scale=(0.7,0.9), p=0.2),  # mild low-res effect
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
                   shadow_dimension=5, p=0.2),
    A.RandomRain(brightness_coefficient=0.95, drop_length=5,
                 drop_width=1, blur_value=2, p=0.1),  # very rare
])

# Background colors (slightly yellowish/white/gray)
bg_colors = [(255,255,255), (250,250,210), (255,255,180), (240,240,240)]

STATE_CODES = [
    "TN", "KA", "MH", "DL", "GJ", "RJ", "UP", "WB", "KL", "AP",
    "TS", "MP", "HR", "PB", "CH", "BR", "JH", "UK", "HP", "GA",
    "OD", "AS", "TR", "MN", "ML", "SK", "AR", "NL", "AN", "LD",
    "PY", "DN", "DD", "JK", "LA"
]

def draw_centered_bold_text(draw, img_size, text, font, fill, y_offset=0):
    w, h = img_size
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x = (w - text_w) // 2
    y = ((h - text_h) // 2) + y_offset
    for dx, dy in [(0,0), (1,0), (0,1), (1,1)]:  # fake bold
        draw.text((x+dx, y+dy), text, font=font, fill=fill)

def make_plate(text, w=512, h=96, two_line=False):
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
        font1 = ImageFont.truetype(random.choice(FONT_PATHS), random.randint(42, 52))
        font2 = ImageFont.truetype(random.choice(FONT_PATHS), random.randint(42, 52))
        draw_centered_bold_text(draw, (w, h), parts[0], font1, text_color, y_offset=-20)
        draw_centered_bold_text(draw, (w, h), parts[1], font2, text_color, y_offset=20)
    else:
        font = ImageFont.truetype(random.choice(FONT_PATHS), random.randint(48, 58))
        draw_centered_bold_text(draw, (w, h), text, font, text_color)

    img = np.array(img)
    img = augment(image=img)["image"]
    return img

def rand_plate_text():
    st = random.choice(STATE_CODES)
    district = f"{random.randint(1,99):02d}"
    series = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=random.choice([1,2])))
    number = f"{random.randint(0,9999):04d}"
    return f"{st}{district}{series}{number}"

# --- Generate plates and CSV ---
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","plate_text"])
    
    for i in range(1, 50):
        text = rand_plate_text()
        two_line = random.random() < 0.3
        img = make_plate(text, two_line=two_line)
        filename = f"plate_{i:05d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.writerow([filename, text])
        if i % 500 == 0:
            print(f"✅ Generated {i}/5000 plates")
