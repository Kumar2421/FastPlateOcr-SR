import os, cv2, math, random, numpy as np, pandas as pd
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import editdistance
from visualize_result import visualize_predictions , plot_training_curves
from model import FastPlateOCR   # <-- your model.py
# ---------------------------
# Repro + cuDNN speed
# ---------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# ---------------------------
# Utility: resize & pad
# ---------------------------
def resize_keep_h_pad(img, target_h, max_w):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    if new_w > max_w:
        new_w = max_w
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.full((target_h, max_w, 3), 255, dtype=np.uint8)
    canvas[:, :new_w] = img
    pil = Image.fromarray(canvas)
    return pil, new_w


# ---------------------------
# Dataset
# ---------------------------
class PlateDataset(Dataset):
    def __init__(self, csv_file, vocab, target_h=96, max_w=512, train=True, img_dir="src2/resized_plates"):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for i, c in enumerate(vocab)}
        self.target_h, self.max_w = target_h, max_w
        self.train = train
        self.img_dir = img_dir

        self.tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.df)

    def text_to_seq(self, text):
        return torch.tensor(
            [self.char2idx['<sos>']] +
            [self.char2idx[c] for c in str(text) if c in self.char2idx] +
            [self.char2idx['<eos>']], dtype=torch.long
        )

    def seq_to_text(self, seq):
        out=[]
        for i in seq:
            i=int(i)
            if i==self.char2idx['<eos>']: break
            if i not in (self.char2idx['<sos>'], self.char2idx['<pad>']):
                out.append(self.idx2char[i])
        return "".join(out)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Missing: {img_path}")
            img = np.zeros((self.target_h, self.max_w, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil, _ = resize_keep_h_pad(img, self.target_h, self.max_w)
        img_t = self.tx(pil)

        text = row["license_plate"]
        seq = self.text_to_seq(text)

        if idx % 500 == 0:
            print(f"📸 Processing {img_path} -> {text}")
        return img_t, seq, text


def collate_fn(batch):
    imgs, seqs, texts = zip(*batch)
    imgs = torch.stack(imgs)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    return imgs, seqs, texts


# ---------------------------
# Metrics
# ---------------------------
def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(1, len(ref))

@torch.no_grad()
def validate(model, loader, dataset, device, beam_width=5):
    model.eval()
    tot_cer, tot_exact, n = 0.0, 0, 0
    for imgs, _, texts in loader:
        imgs = imgs.to(device)
        for i in range(imgs.size(0)):
            ids = model.beam_decode(imgs[i].unsqueeze(0), beam_width=beam_width, device=device)
            pred = dataset.seq_to_text(ids)
            gt = texts[i]
            tot_cer += cer(gt, pred)
            tot_exact += (gt == pred)
            n += 1
    return (tot_cer/n, tot_exact/n) if n>0 else (0.0,0.0)


# ---------------------------
# Training function
# ---------------------------
def train_fastplateocr(csv_file, img_dir="src2/resized_plates",
                       epochs=20, batch_size=16, beam_width=5, lr=1e-4):

    # vocab
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    vocab = ['<pad>', '<sos>', '<eos>'] + list(charset)

    # dataset
    dataset = PlateDataset(csv_file, vocab, img_dir=img_dir, train=True)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, collate_fn=collate_fn, num_workers=1)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastPlateOCR(vocab_size=len(vocab)).to(device)

    # optimizer + scheduler + scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # training logs
    logs = {
        "train_losses": [],
        "val_cers": [],
        "val_accs": []
    }
    train_losses, val_cers, val_accs = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, seqs, _ in train_loader:
            imgs, seqs = imgs.to(device), seqs.to(device)
            tgt_inp, tgt_out = seqs[:, :-1], seqs[:, 1:]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs, tgt_inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=0
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_cer, val_acc = validate(model, val_loader, dataset, device, beam_width)
        val_cers.append(val_cer); val_accs.append(val_acc)
        # 🔥 Save sample result images each epoch
        # Update charts
        logs["train_losses"].append(total_loss/len(train_loader))
        logs["val_cers"].append(val_cer)
        logs["val_accs"].append(val_acc)
        plot_training_curves(logs, out_dir="results")
        visualize_predictions(model, val_loader, dataset, device, epoch, out_dir="results")

        print(f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | CER {val_cer:.3f} | Acc {val_acc:.3f} | lr {scheduler.get_last_lr()[0]:.6f}")

    # ---------------------------
    # Plot results
    # ---------------------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.plot(train_losses); plt.title("Train Loss")
    plt.subplot(1,3,2); plt.plot(val_cers); plt.title("Val CER")
    plt.subplot(1,3,3); plt.plot(val_accs); plt.title("Val Acc")
    plt.tight_layout()
    plt.savefig("training_curves.png"); plt.close()

    return model, vocab, (train_losses, val_cers, val_accs)


def visualize_predictions(model, dataset, device, epoch, num_samples=6, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idxs = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for j, idx in enumerate(idxs):
            img_t, _, gt_text = dataset[idx]
            img_t = img_t.unsqueeze(0).to(device)

            ids = model.beam_decode(img_t, beam_width=5, device=device)
            pred_text = dataset.seq_to_text(ids)

            # Convert back to numpy for visualization
            img = img_t[0].cpu().permute(1,2,0).numpy()
            img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
            img = np.clip(img, 0, 1)

            axes[j].imshow(img)
            axes[j].set_title(f"GT: {gt_text}\nPred: {pred_text}", fontsize=9)
            axes[j].axis("off")

        # Hide unused subplots
        for k in range(j+1, len(axes)):
            axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch:02d}.png"))
    plt.close()


# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    model, vocab, logs = train_fastplateocr("image_data.csv",
                                            img_dir="resized_plates",
                                            epochs=20,
                                            batch_size=32,
                                            beam_width=5)
