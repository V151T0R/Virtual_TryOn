# 🧥 Multi U-Net Based Virtual Try-On System

A **resource-efficient virtual try-on (VTON) framework** based on a **Multi U-Net architecture with Self-Excitation (SE) blocks** for realistic image-to-image garment fitting.  
The system balances **structural alignment**, **identity preservation**, and **computational efficiency**, making it suitable for real-world deployment on limited hardware.

---

## 📌 Overview

Virtual Try-On (VTON) systems allow users to visualize garments on human images before purchase, reducing return rates and improving customer experience in e-commerce.

This project proposes a **two-stage U-Net pipeline**:
1. **Segmentation U-Net** – Generates an agnostic person representation.
2. **Translational U-Net** – Synthesizes the final try-on image by fusing garment and person features using **Self-Excitation blocks**.

The model is trained and evaluated on the **VITON-HD dataset**.

---

## 🧠 Key Contributions

- Multi U-Net based virtual try-on architecture  
- Self-Excitation (SE) blocks for channel-wise feature recalibration  
- End-to-end image-to-image translation (no explicit warping module)  
- Lightweight and computationally efficient  
- Evaluated using SSIM, LPIPS, and FID  

---

## 🏗️ Architecture

### Core Components

- **U-Net Encoder–Decoder**
  - Preserves spatial details via skip connections
- **Self-Excitation Blocks (SEB)**
  - Emphasize garment- and body-relevant channels
- **Fusion Block**
  - Concatenates segmented person image and garment image
- **Two-Stage Pipeline**
  - Segmentation → Translation

---

## 🔄 VT-ON Pipeline

