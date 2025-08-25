import cv2
import torch
from torchvision import transforms
from dataset import PlateDataset, resize_keep_h_pad
from model import FastPlateOCR


# ---------------------------
# Inference helper (single image)
# ---------------------------
@torch.no_grad()
def infer_single(img_path, model, vocab, device, beam_width=5, target_h=96, max_w=512):
    ds_dummy = PlateDataset.__new__(PlateDataset)  # tiny trick to reuse seq_to_text
    ds_dummy.vocab=vocab
    ds_dummy.char2idx = {c:i for i,c in enumerate(vocab)}
    ds_dummy.idx2char = {i:c for i,c in enumerate(vocab)}
    # prep
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    pil,_ = resize_keep_h_pad(img, target_h=target_h, max_w=max_w)
    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_t = tx(pil).to(device)
    ids = model.beam_decode(img_t, beam_width=beam_width, device=device)
    pred = ds_dummy.seq_to_text(ids)
    return pred