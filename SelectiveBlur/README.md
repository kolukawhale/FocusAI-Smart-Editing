# Selective Region Blur – README

## 1. Introduction
This project implements a selective region blurring system capable of applying privacy-preserving blur to user-defined areas of an image. Users draw a closed boundary around sensitive content, and the system automatically fills the enclosed region and applies a blur of adjustable intensity (mild, medium, strong).

The primary goal was to build a technically strong image-processing pipeline that delivers high-quality, unrecognizable blur for privacy protection, while maintaining a strict processing-time requirement (<25 seconds).

To achieve this, we designed a hybrid approach combining a lightweight convolutional blurring network (BlurNet) with optimized classical image-processing operations, ultimately achieving both strong intensity and practical performance.

## 2. System Architecture
```
                       ┌────────────────────────────┐
                       │        User Input           │
                       │  (Image + Drawn Boundary)   │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                     ┌──────────────────────────────────┐
                     │     Mask Extraction Module        │
                     │  - Extract boundary strokes       │
                     │  - Convert to binary mask         │
                     │  - Auto-fill the enclosed region  │
                     └──────────────────┬────────────────┘
                                        │
                                        ▼
                          ┌────────────────────────┐
                          │   Feathering Module    │
                          │ - Smooth mask edges    │
                          └──────────┬─────────────┘
                                     │
                                     ▼
          ┌──────────────────────────────────────────────────────────┐
          │                       BlurNet (CNN)                      │
          │  - Multi-layer convolutional smoothing                   │
          │  - Gaussian-initialized kernels                          │
          │  - Configurable mild / medium / strong settings          │
          └─────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
             ┌───────────────────────────────────────────────────┐
             │      Unified Classical Enhancement Pipeline       │
             │   - Gaussian blur (large kernel)                  │
             │   - Box blur refinement                           │
             └──────────────────────┬────────────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │    Final Image Composition     │
                    │ - Blend original + blurred     │
                    │   using feathered mask         │
                    └────────────────────────────────┘
```

## 3. Method

### 3.1 BlurNet: CNN-Based Smoothing
BlurNet is a lightweight convolutional neural network used to generate an initial layer of smoothing:
- Depthwise convolution layers
- Gaussian-initialized kernels
- Progressively increasing sigma
- Configurations:
  - Mild: 3 layers, σ=3.0
  - Medium: 5 layers, σ=5.0
  - Strong: 7 layers, σ=7.0

### 3.2 Classical Blur Enhancement
To achieve extremely strong blur suitable for privacy masking, we applied:
- GaussianBlur with large kernels  
- BoxBlur for diffusive smoothing  

These steps amplify BlurNet’s output and allow full anonymization.

### 3.3 Mask Processing
- Convert user-drawn strokes into binary mask  
- Fill enclosed region  
- Apply Gaussian feathering for smooth transitions  
- Blend blurred region naturally with original image  

### 3.4 Performance Optimization
The pipeline was carefully optimized:
- Images resized to a maximum dimension of 1200px  
- Removed triple-pass CNN convolutions  
- Reduced CNN kernel sizes  
- Unified blur pipeline  
- Minimized Python overhead  

This enabled keeping execution under **25 seconds** even for strong blur.

---

## 4. What Didn’t Work

### 4.1 Heavy CNN Architecture
An initial pure-CNN approach attempted:

```
mild:   5 layers, σ=5.0, kernel=11  
medium: 7 layers, σ=9.0, kernel=17  
strong: 10 layers, σ=14.0, kernel=25  
```

Issues:
- Processing time exceeded **90 seconds**  
- CPU inference too slow for such large convolutions  
- Strong blur required multiple stacked passes  
- Failed to meet project constraints  

### 4.2 Triple-Pass Convolution
We initially applied:
```
x = layer(x)
x = layer(x)
x = layer(x)
```
per layer.

This produced strong blur but increased runtime by ~3×.

Both approaches were discarded.

---

## 5. What Worked (Final Approach)

### Hybrid CNN + Classical Blur Pipeline
The final system combines:
- BlurNet for structure-aware smoothing  
- OpenCV Gaussian + Box blur for high-intensity anonymization  

### Blur intensities:
- **Mild:** Gaussian(25) + Box(15)
- **Medium:** Gaussian(45) + Box(25)
- **Strong:** Gaussian(65) + Box(45)

This provides:
- Strong, unrecognizable anonymization at high setting  
- Natural, smooth blending  
- Runtime < 25 seconds  

---

## 6. Key Learnings

### 1. Pure CNN blur is too slow on CPU  
Deep CNNs with large kernels take too long without GPU acceleration.

### 2. CNNs can be effective without training  
BlurNet uses analytically initialized Gaussian kernels — fast and predictable.

### 3. Hybrid systems are more powerful  
A combination of CNN + classical image processing outperforms either alone.

### 4. Mask feathering is essential  
Without feathering, boundaries appear unnatural. Gaussian feathering solves this.

### 5. Engineering trade-offs matter  
The final system favors real performance and user experience over deep architectures.

---

## 7. Final Summary
This project delivers a robust, high-performance selective blurring system capable of anonymizing sensitive regions in images with adjustable intensity. The hybrid design (BlurNet + classical enhancement) ensures strong privacy protection, natural transitions, and reliable execution under 25 seconds. This makes the system practical and technically sound for real-world applications such as privacy filtering, content moderation, and preprocessing in vision pipelines.

