import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, C]
        return x + self.pe[:, :x.size(1)]


# ---------------------------
# FastPlateOCR Model
# ---------------------------
class FastPlateOCR(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, pretrained=True):
        super().__init__()

        # CNN encoder backbone
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.cnn = nn.Sequential(*list(backbone.features.children()))
        cnn_out = 576  # MobileNetV3-small last channel size

        # projection to transformer dim
        self.proj = nn.Conv2d(cnn_out, d_model, 1)

        # transformer decoder
        self.pos_enc = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=512, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # embeddings + classifier
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, imgs, tgt_inp):
        # imgs: [B,3,H,W], tgt_inp: [B,T]
        feats = self.cnn(imgs)                  # [B,C,H’,W’]
        feats = self.proj(feats)                # [B,d,H’,W’]
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B, H*W, C).permute(1, 0, 2)

        tgt_emb = self.embedding(tgt_inp) * (self.d_model ** 0.5)
        tgt_emb = self.pos_enc(tgt_emb)        # add PE
        tgt_emb = tgt_emb.permute(1, 0, 2)

        out = self.decoder(tgt_emb, feats)     # [T,B,C]
        out = out.permute(1, 0, 2)
        return self.fc_out(out)

    # ---------------------------
    # Greedy decode
    # ---------------------------
    def greedy_decode(self, img, max_len=32, sos_id=1, eos_id=2, device="cpu"):
        self.eval()
        with torch.no_grad():
            feats = self.cnn(img)
            feats = self.proj(feats)
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(B, H*W, C).permute(1, 0, 2)

            ys = torch.full((1, 1), sos_id, dtype=torch.long, device=device)
            for _ in range(max_len - 1):
                tgt_emb = self.embedding(ys) * (self.d_model ** 0.5)
                tgt_emb = self.pos_enc(tgt_emb)
                tgt_emb = tgt_emb.permute(1, 0, 2)
                out = self.decoder(tgt_emb, feats)
                out = self.fc_out(out.permute(1, 0, 2))[:, -1, :]
                next_word = out.argmax(dim=-1).unsqueeze(0)
                ys = torch.cat([ys, next_word], dim=1)
                if next_word.item() == eos_id:
                    break
        return ys.squeeze(0).tolist()

    # ---------------------------
    # Beam search (memory-safe)
    # ---------------------------
    def beam_decode(self, img, beam_width=5, max_len=32, sos_id=1, eos_id=2, device="cpu"):
        self.eval()
        with torch.no_grad():
            feats = self.cnn(img)
            feats = self.proj(feats)
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(B, H*W, C).permute(1, 0, 2)

            # [seq, score]
            sequences = [(torch.full((1, 1), sos_id, dtype=torch.long, device=device), 0.0)]

            for _ in range(max_len):
                all_candidates = []
                for seq, score in sequences:
                    if seq[0, -1].item() == eos_id:
                        all_candidates.append((seq, score))
                        continue

                    tgt_emb = self.embedding(seq) * (self.d_model ** 0.5)
                    tgt_emb = self.pos_enc(tgt_emb)
                    tgt_emb = tgt_emb.permute(1, 0, 2)
                    out = self.decoder(tgt_emb, feats)
                    out = self.fc_out(out.permute(1, 0, 2))[:, -1, :]
                    log_probs = F.log_softmax(out, dim=-1)

                    topk = torch.topk(log_probs, beam_width)
                    for i in range(beam_width):
                        token = topk.indices[0, i].item()
                        prob = topk.values[0, i].item()
                        new_seq = torch.cat(
                            [seq, torch.tensor([[token]], device=device)], dim=1
                        )
                        all_candidates.append((new_seq, score + prob))

                sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

            return sequences[0][0].squeeze(0).tolist()
