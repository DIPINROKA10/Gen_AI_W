# Assignment 6: Applying AI-Based Style Transfer to an Image

## Objective
Compare two different methods of image style transfer: CycleGAN and Neural Style Transfer (VGG-based).

---

## Images

### 1. Original Image
![Original Image](original.jpg)

### 2. CycleGAN Output
![CycleGAN Output](cyclegan_output.jpg)

### 3. Neural Style Transfer Output
![Neural Style Transfer Output](nst_output.jpg)

### Side-by-Side Comparison
![Comparison](comparison.png)

---

## Written Comparison

**Content Preservation:** Neural Style Transfer better preserves the original content structure, maintaining the spatial layout and key features of the landscape while applying artistic style. CycleGAN tends to modify structural elements more aggressively due to its adversarial training objective.

**Style Transformation Strength:** CycleGAN produces a stronger, more dramatic style transformation as it learns a complete domain mapping (e.g., photo → painting) through adversarial loss. Neural Style Transfer applies style more subtly, blending content and style statistics via Gram matrices.

**Key Differences:** 
- **CycleGAN** uses cycle consistency loss (forward-backward reconstruction: G(F(x)) ≈ x) to ensure the transformation is invertible and preserves content identity across domains. This bidirectional constraint helps maintain structural coherence but can limit style expressiveness.
- **Neural Style Transfer** optimizes a single image to match content features (from VGG conv layers) and style features (Gram matrices of VGG activations) simultaneously, without requiring paired training data or cycle constraints.
- CycleGAN requires pretrained models for specific domain pairs (e.g., horse↔zebra, monet↔photo), while NST works with any style image instantly.

---

## Technical Notes

**Cycle Consistency in CycleGAN:** The cycle consistency loss (L_cyc) enforces that translating an image from domain A to B and back to A should recover the original: ||G(F(x)) - x||₁. This prevents mode collapse and ensures the generator learns meaningful mappings rather than arbitrary transformations.

**Neural Style Transfer:** Uses VGG-19 features. Content loss compares feature maps at deeper layers (conv4_2), while style loss compares Gram matrices (channel correlations) at multiple layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1) to capture texture patterns.