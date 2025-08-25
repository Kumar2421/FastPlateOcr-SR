# utils.py
import os
import cv2
import torch
import numpy as np
import random
import editdistance
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# Resize + pad (right-side padding)
# ---------------------------
def resize_keep_h_pad(img_rgb, target_h=96, max_w=512, pad_value=128):
    h0, w0 = img_rgb.shape[:2]
    scale = target_h / float(h0)
    new_w = max(1, min(max_w, int(round(w0 * scale))))
    img = cv2.resize(img_rgb, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    if new_w < max_w:
        pad = np.ones((target_h, max_w - new_w, 3), dtype=np.uint8) * pad_value
        img = np.concatenate([img, pad], axis=1)
    return Image.fromarray(img), new_w

# ---------------------------
# Collate function for dataloader
# ---------------------------
def collate_fn(batch):
    imgs, seqs, texts = zip(*batch)
    imgs = torch.stack(imgs)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    return imgs, seqs, texts

# ---------------------------
# Evaluation metrics
# ---------------------------
def cer(ref, hyp):
    """Character Error Rate"""
    return editdistance.eval(ref, hyp) / max(1, len(ref))

def wer(ref, hyp):
    """Word Error Rate"""
    ref_words = ref.split()
    hyp_words = hyp.split()
    return editdistance.eval(ref_words, hyp_words) / max(1, len(ref_words))

# ---------------------------
# Inference helpers
# ---------------------------
@torch.no_grad()
def infer_single(img_path, model, vocab, device, beam_width=5, target_h=96, max_w=512):
    """
    Inference on a single image path.
    """
    from build_arc import PlateDataset  # lazy import to avoid circular
    ds_dummy = PlateDataset.__new__(PlateDataset)
    ds_dummy.vocab = vocab
    ds_dummy.char2idx = {c: i for i, c in enumerate(vocab)}
    ds_dummy.idx2char = {i: c for i, c in enumerate(vocab)}

    # prep
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    pil, _ = resize_keep_h_pad(img, target_h=target_h, max_w=max_w)
    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_t = tx(pil).to(device)

    # beam decode
    ids = model.beam_decode(img_t, beam_width=beam_width, device=device)
    pred = ds_dummy.seq_to_text(ids)
    return pred

@torch.no_grad()
def infer_folder(img_dir, model, vocab, device, beam_width=5):
    """
    Inference on all images inside a folder.
    """
    results = {}
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            fpath = os.path.join(img_dir, fname)
            pred = infer_single(fpath, model, vocab, device, beam_width=beam_width)
            results[fname] = pred
            print(f"📸 {fname} -> {pred}")
    return results

# ---------------------------
# EMA (for stable training)
# ---------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v * (1 - self.decay))

    def store(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}

    def load(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=False)
