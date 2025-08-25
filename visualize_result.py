import os, matplotlib.pyplot as plt
import random, numpy as np

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

def plot_training_curves(logs, out_dir="results"):
    """
    Plot training loss, validation CER, and accuracy over epochs.
    logs = {"train_losses": [...], "val_cers": [...], "val_accs": [...]}
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(logs["train_losses"]) + 1)

    plt.figure(figsize=(12,4))

    # Loss curve
    plt.subplot(1,3,1)
    plt.plot(epochs, logs["train_losses"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # CER curve
    plt.subplot(1,3,2)
    plt.plot(epochs, logs["val_cers"], label="Val CER", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.title("Validation CER")
    plt.legend()

    # Accuracy curve
    plt.subplot(1,3,3)
    plt.plot(epochs, logs["val_accs"], label="Val Acc", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curves.png"))
    plt.close()
