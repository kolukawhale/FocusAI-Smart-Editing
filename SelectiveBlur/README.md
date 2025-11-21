# FocusAI Checkpoint 2: Image Denoising with CNNs

## Project Overview

This checkpoint implements a deep learning-based image denoising system using Convolutional Neural Networks (CNNs). The goal is to demonstrate how CNNs can effectively remove noise from degraded images while preserving important details and structures.

## What We Built

**DnCNN (Denoising Convolutional Neural Network):**
- 7-layer CNN architecture with residual learning
- Trained on clean images with synthetically added Gaussian noise
- Learns to predict and remove noise patterns
- Achieves 8-10 dB PSNR improvement on test images

**Key Features:**
- Synthetic noise generation (σ=25 Gaussian noise)
- Multi-stage training with learning rate scheduling
- Validation monitoring to prevent overfitting
- Quantitative evaluation using PSNR and SSIM metrics
- Generalization testing on Fashion MNIST dataset

## Why We Pivoted from Super-Resolution

### Initial Approach: Image Compression with Super-Resolution

**Original Plan:**
- Compress images aggressively (3-4× downsampling, 9-16× storage reduction)
- Use SRCNN (Super-Resolution CNN) to reconstruct high-quality images on-demand
- Enable efficient bulk storage with high-quality retrieval capability

**Why It Didn't Work:**

1. **Fundamental Information Loss:** When images are compressed through downsampling, critical information is permanently destroyed. No algorithm—including deep learning—can truly recover what was lost. The CNN can only make educated guesses based on learned patterns, not reconstruct actual missing details.

2. **High Baseline Performance:** For our test images (1920×1080 compressed to 640×360), bicubic interpolation already achieved 29-32 dB PSNR. This left minimal room for improvement. Our SRCNN achieved 29.31 dB versus bicubic's 29.34 dB—actually 0.03 dB worse!

3. **Image Size Issues:** 
   - High-resolution images (4K, 1080p) remained high-quality even after compression
   - No visible degradation to "fix"
   - Super-resolution only shows benefits when reconstructing from very low resolution (256×256 or less)

4. **Training Challenges:**
   - Training on 4K images was extremely slow (3+ hours)
   - Resizing training images to 512px for speed destroyed the fine details the model needed to learn
   - Model struggled to generalize across different image resolutions

5. **Limited Practical Value:** The visual difference between bicubic upscaling and CNN reconstruction was imperceptible, making it difficult to demonstrate the value of the deep learning approach.

### New Approach: Image Denoising

**Why Denoising Works Better:**

1. **Information Preservation:** Unlike compression, noise addition doesn't destroy information—it corrupts it. The original signal still exists underneath the noise, making recovery feasible.

2. **Clear Problem to Solve:** Noisy images (20-22 dB PSNR) are visibly degraded. The model has an obvious target: remove corruption while preserving structure.

3. **Dramatic Results:** Our DnCNN achieves +8.97 dB improvement (20.63 dB → 29.60 dB), with SSIM improving from 0.247 to 0.700. This is visually obvious and quantitatively impressive.

4. **Stable Training:** Denoising models train faster and more reliably. The task is well-defined: learn noise patterns and subtract them.

5. **Real-World Applications:** 
   - Cleaning up photos taken in low light (high ISO noise)
   - Restoring old/degraded photographs
   - Removing JPEG compression artifacts
   - Pre-processing for other computer vision tasks

6. **Better Generalization:** Our model trained on natural images successfully denoises Fashion MNIST clothing sketches, demonstrating it learned general noise removal rather than dataset-specific patterns.

## Technical Implementation

### Architecture: DnCNN
```
Input (Noisy RGB Image)
    ↓
Conv2d (3→64, 3×3) + ReLU
    ↓
5× [Conv2d (64→64, 3×3) + BatchNorm + ReLU]
    ↓
Conv2d (64→3, 3×3)
    ↓
Predicted Noise
    ↓
Output = Input - Predicted Noise (Residual Learning)
```

### Training Configuration
- **Dataset:** 20 high-quality images (preprocessed to 800×800)
- **Noise Level:** σ=25 (Gaussian noise)
- **Patch Size:** 64×64
- **Batch Size:** 16
- **Epochs:** 30
- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE (Mean Squared Error)
- **Training Time:** ~30-40 minutes on GPU

### Results

**Natural Images:**
- Input (Noisy): 20.63 dB PSNR, 0.247 SSIM
- Output (Denoised): 29.60 dB PSNR, 0.700 SSIM
- **Improvement: +8.97 dB, +0.453 SSIM**

**Fashion MNIST (Generalization Test):**
- Successfully denoises clothing sketches despite training only on natural images
- Demonstrates learned denoising is general-purpose, not dataset-specific

## Key Learnings

1. **Deep learning cannot create information that was destroyed** (super-resolution limitation)
2. **Deep learning excels at removing corruption from existing information** (denoising success)
3. **Choosing the right problem is as important as the solution** - pivoting to denoising gave us visually impressive, quantitatively strong results
4. **Residual learning is effective** - predicting noise rather than clean image directly improves training
5. **Models can generalize beyond training distribution** - natural image training → Fashion MNIST success

## Files Included

- `Denoising.ipynb` - Complete training pipeline
- `results/` - Sample denoised images and visualizations
- Training loss curves and metric plots

## Future Work

- Train with variable noise levels (σ=15-50) for robustness
- Implement real-world noise models (Poisson, speckle)
- Extend to JPEG artifact removal
- Integrate with FocusAI's region-based editing for selective denoising
- Explore blind denoising (unknown noise levels)

## Conclusion

While super-resolution revealed fundamental limitations of reconstructing lost information, pivoting to image denoising demonstrated the power of CNNs for corruption removal. The +9 dB improvement and successful generalization to unseen data validate that CNNs can effectively learn to separate signal from noise, a capability with broad practical applications in image restoration and preprocessing.