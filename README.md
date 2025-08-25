# 🚘 FastPlateOCR-Pro

An efficient and robust **Automatic License Plate OCR model** optimized for real-world conditions (blurry, low-res, noisy plates).  
This project is an improved version of **FastPlateOCR**, with the following upgrades:

✅ Super-Resolution (SR) preprocessing for tiny/blurred plates  
✅ Beam Search decoding for higher accuracy  
✅ Mixed Precision training (AMP) for faster GPU usage  
✅ EMA (Exponential Moving Average) stabilization  
✅ Clean dataset pipeline (CSV + folder based)  
✅ Runs fully on GPU (tested on RTX 3050 4GB)  
✅ Supports both **images and video**  

---

## 📂 Dataset Format

CSV file required:
```csv
image_name,license_plate
image_1.jpg,TN56P0837
image_2.jpg,TN77T4062
...
