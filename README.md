FastPlateOcr-SR 🚗🔍

Lightweight and accurate Automatic License Plate Recognition (ALPR) system built with PyTorch.
It combines:

> CNN encoder (MobileNetV3-Small) for fast feature extraction

> Transformer decoder for sequence prediction

> SRLite module (tiny super-resolution residual network) for denoising & enhancing plates before OCR

> Beam search decoding for higher accuracy

> CUDA AMP (mixed precision) for faster GPU training

🔧 Features

> End-to-end trainable license plate OCR

> Supports custom datasets (CSV + images)

> Data augmentation (blur, jitter, affine, downscale-upscale, JPEG artifacts)

> Greedy / Beam Search decoding

> Super lightweight → suitable for real-time inference

> Works with CUDA 12.x


project structure:

FastPlateOcr-SR/
│── build_arc.py      # Training script & entry point
│── model.py          # FastPlateOCR model + SRLite block
│── utils.py          # Dataset, transforms, metrics
│── requirements.txt  # Dependencies
│── image_data.csv    # Example CSV (image_name, license_plate)
│── resized_plates/   # Training images (preprocessed)
│── README.md         # This file


📊 Dataset Format

Your dataset CSV should look like this:
image_name,license_plate
image_1.jpg,TN56P0837
image_2.jpg,TN77T4062
image_3.jpg,TN70AP9285


🚀 Installation

1.Clone the repo
git clone https://github.com/Kumar2421/FastPlateOcr-SR.git
cd FastPlateOcr-SR

2.Create environment & install dependencies (for CUDA 12.7)
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

🏋️ Training

Run:
python build_arc.py \
  --csv image_data.csv \
  --img_dir resized_plates \
  --epochs 20 \
  --batch_size 32 \
  --beam_width 5


During training, you’ll see:
>Loss
>Character Error Rate (CER)
>Accuracy
>Learning rate
>At the end, logs are saved and can be plotted.

📈 Visualizing Results

After training, plot loss & accuracy curves:

from utils import plot_training_logs

plot_training_logs(logs, save_path="training_curves.png")

🔍 Inference Example

import torch
from model import FastPlateOCR
from utils import PlateDataset

# Load vocab + model
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + ["<pad>", "<sos>", "<eos>"]
model = FastPlateOCR(vocab_size=len(vocab))
model.load_state_dict(torch.load("fastplateocr_best.pth"))
model.eval()

# Run inference on single image
ids = model.beam_decode(img_tensor, beam_width=5, device="cuda")
pred_text = dataset.seq_to_text(ids)
print("Predicted plate:", pred_text)


⚡ Results (example)
Epoch	Loss	CER	Accuracy
1	2.389	1.626	0.0%
20	0.214	0.053	97.5%

(update with your actual results)

📜 Citation

@article{fastplateocr-sr,
  author = {Senthil kumar},
  organization = {fusion apps},
  title = {FastPlateOcr-SR: Lightweight Transformer-based License Plate Recognition},
  year = {2025},
  howpublished = {\url{https://github.com/Kumar2421/FastPlateOcr-SR}}
}
