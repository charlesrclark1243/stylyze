# stylyze

**Author:** Charlie Clark \
**Date Started:** 2026-09-12

## Contents

1. [What is stylyze](#what-is-stylyze)
2. [How does stylyze work?](#how-does-stylyze-work)
   - [The idea](#the-idea)
   - [Architecture](#architecture)
   - [Style strength](#style-strength)
   - [Training](#training)
   - [Serving it](#serving-it)

## What is stylyze?

stylyze is a neural style transfer (NST) application. The user uploads both a real-world photograph (the content image) and a painting (the style image). Using a deep neural architecture, stylyze will transfer the artistic style from the uploaded painting to the user's real-world photograph. stylyze can do this without ever having previously seen either the content image or the style image.

## How does stylyze work?

stylyze is an implementation of [Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
(Huang & Belongie, 2017), trained from scratch on COCO and WikiArt.

### The idea

A convolutional network trained on natural images ends up storing style — palette, brush
texture, contrast — in the *statistics* of its feature maps, while content — what is actually
depicted — lives in their spatial arrangement. AdaIN uses this directly: normalize the content
image's features to zero mean and unit variance per channel, then rescale them using the mean
and standard deviation of the style image's features. The arrangement survives, the statistics
are replaced.

That operation has no learned parameters, which is what makes the style *arbitrary*. Earlier
fast style transfer methods trained one network per style; stylyze has never seen your painting
before and does not need to. The style image is an input, not a weight.

### Architecture

```
content ─┐
         ├─► VGG-19 encoder ─► AdaIN ─► decoder ─► stylized image
style  ──┘     (frozen)      (no params)  (trained)
```

**Encoder.** ImageNet-pretrained VGG-19, truncated at `relu4_1`. Frozen throughout: it is a
fixed measuring instrument, not something being learned.

**AdaIN.** Normalizes content features and rescales them with the style's per-channel mean and
standard deviation. Parameter-free.

**Decoder.** Mirrors the encoder in reverse — 3×3 convolutions with reflection padding, three
nearest-neighbour 2× upsamples, and residual blocks at 512, 256 and 128 channels. Reflection
padding avoids the border artifacts that zero padding leaves around the frame, and the output
is deliberately unbounded (no sigmoid), so activations cannot saturate during training.

Only the decoder is trained. The task it learns is narrow and well-posed: invert AdaIN's output
back into an image.

### Style strength

The `alpha` slider interpolates in *feature space*, before decoding:

```
target = alpha · AdaIN(content, style) + (1 - alpha) · content_features
```

At `alpha = 0` the decoder reconstructs the photograph, at `1` it applies the style fully, and
in between it produces genuinely intermediate stylizations. Blending the decoded pixels instead
would merely cross-fade two finished images, which looks like a dissolve rather than a weaker
style.

### Training

| | |
|---|---|
| Content images | COCO `train2017` — 118,287 photographs |
| Style images | WikiArt — 65,166 paintings, randomly paired with content |
| Preprocessing | shorter side resized to 512, random 256×256 crop |
| Optimizer | Adam, learning rate 1e-4 (1e-3 makes activations explode within a few steps) |
| Batch size | 8 |
| Epochs | 30 |
| Trained parameters | the decoder only |

Four losses, all measured through the same frozen encoder:

- **Content** (λ=1) — MSE between the output's `relu4_1` features and the AdaIN target. Note the
  target is AdaIN's output, not the content image's features: the decoder is being taught to
  invert AdaIN, not to reproduce the photograph.
- **Style** (λ=10) — MSE on per-channel means and standard deviations at `relu1_1` through
  `relu4_1`. Matching statistics at four depths captures both fine texture and broad palette.
- **Identity** (λ=1) — L1 reconstruction loss for stylizing an image with *itself*, which should
  return it unchanged. A cheap, strong signal that keeps the decoder honest.
- **Total variation** (λ=1) — penalizes differences between neighbouring pixels, suppressing the
  checkerboard artifacts that upsampling tends to produce.

### Serving it

The trained model is exported to ONNX and served with ONNX Runtime on CPU, which is about 20%
faster than PyTorch here and shrinks the backend image from 1.6 GB to 684 MB — the serving
container needs no deep learning framework at all, just a 66 MB graph.

A FastAPI backend and an Angular frontend run behind nginx and Caddy on a single EC2 instance.
Uploads are resized to a 512-pixel shorter side and cropped to a multiple of 8, since the
decoder's three 2× upsamples fix the output to that grid. One image takes roughly 10 seconds on
two vCPUs, so the API serves a single inference at a time and sheds bursts rather than queueing
them indefinitely.
