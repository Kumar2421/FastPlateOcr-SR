# infer_fastplateocr.py
import os, cv2, torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import FastPlateOCR   # <-- your architecture

# ---------------------------
# Config
# ---------------------------
CHECKPOINT = "checkpoints1/best.pth"
VOCAB = ['<pad>', '<sos>', '<eos>'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Image Preprocessing
# ---------------------------
def resize_keep_h_pad(img, target_h=96, max_w=512):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    if new_w > max_w: new_w = max_w
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.full((target_h, max_w, 3), 255, dtype=np.uint8)
    canvas[:, :new_w] = img
    return Image.fromarray(canvas)

tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------------------
# Helpers
# ---------------------------
def seq_to_text(seq, idx2char):
    out=[]
    for i in seq:
        i=int(i)
        if i==0: continue              # <pad>
        if idx2char[i] in ["<sos>", "<eos>"]: continue
        out.append(idx2char[i])
    return "".join(out)

# ---------------------------
# Load model
# ---------------------------
def load_model():
    model = FastPlateOCR(vocab_size=len(VOCAB)).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

# ---------------------------
# Run Inference
# ---------------------------
@torch.no_grad()
def recognize_plate(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = resize_keep_h_pad(img, 96, 512)
    img_t = tx(pil).unsqueeze(0).to(device)

    ids = model.beam_decode(img_t, beam_width=5, device=device)
    idx2char = {i:c for i,c in enumerate(VOCAB)}
    text = seq_to_text(ids, idx2char)
    return text

# ---------------------------
# Example
# ---------------------------
if __name__ == "__main__":
    model = load_model()
    test_img = "synthetic_plates2/plate_00007.jpg"
    pred = recognize_plate(test_img, model)
    print(f"Predicted Plate: {pred}")
