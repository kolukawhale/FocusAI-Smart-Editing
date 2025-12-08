# FocusAI – Smart Editing

A collection of Deep Learning–based image editing tools built to explore how AI can enhance, restore, and anonymize photos.
Each team member approached photo editing from a different angle, so instead of building one model, we built three independent—but complementary—modules:

1. Image Denoising (DnCNN)

2. Image Enhancement (Two DL Pipelines)

3. Selective Region Blur (Privacy Anonymization)

Our long-term goal is to combine all of these into a **unified web platform for end-to-end intelligent photo editing**.

## Why We Built This

Photo editing covers a huge range of real-world needs—from restoring noisy images, to improving aesthetics, to protecting privacy.
Since each of us had a different idea of what Deep Learning could solve, we decided to tackle multiple ideas at once and explore:

### How far can Deep Learning go in providing a complete image-editing workflow?

This repository documents our experiments, models, and results across all three directions.

## What We Built
### 1. Image Denoising using CNNs (DnCNN)

We explored noise removal using a 7-layer DnCNN that predicts and subtracts noise from corrupted images.
Result: Achieved +9 dB PSNR improvement and strong generalization across domains. 

### 2. Image Enhancement (Two Independent Approaches)

We attempted two enhancement pipelines based on the MIT-Adobe FiveK dataset:

#### Patch-based residual CNN (local detail enhancement)

#### Full-image enhancement with perceptual + color losses (global tone correction)
Result: Both models produced meaningful enhancement with different strengths—one capturing fine details, the other achieving better global color balance. 


### 3. Selective Region Blur (Privacy Anonymization)

A region-based blurring system using a hybrid pipeline:
CNN (BlurNet) + classical Gaussian blurring + mask feathering.
Built with a Gradio interface, allowing users to draw a region to anonymize.
Result: High-quality, unrecognizable blur in under 25 seconds, with smooth natural blending.

## Future Scope

Integrate all modules into a single web application.

Support real-time previews and batch processing.

Optimize CNN components for deployment.

