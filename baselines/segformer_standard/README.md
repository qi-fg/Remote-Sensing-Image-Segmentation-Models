# Standard SegFormer-B0

This baseline is intended for **full-supervision reference / benchmark** use.

- architecture: standard SegFormer-B0 (MiT-B0 + all-MLP decode head);
- encoder initialization: NVIDIA `nvidia/mit-b0`, pretrained on ImageNet-1K;
- implementation: Hugging Face Transformers SegFormer;
- input: RGB tensors in `[0, 1]`; ImageNet normalization is applied inside the wrapper;
- downstream segmentation head: randomly initialized for the requested number of classes.

Install the optional dependency:

```bash
pip install -r baselines/segformer_standard/requirements.txt
```

LoveDA debug example:

```bash
python train.py \
  --dataset loveda \
  --model segformer_b0 \
  --mode debug \
  --data /private/SEG/LoveDA \
  --device cuda
```

The first pretrained run downloads/caches `nvidia/mit-b0`. After the cache is
available, later runs can reuse it.

## Important comparison note

This is a **prompt-free, fully supervised semantic segmentation baseline**. It
must not be relabeled as the prompt-conditioned `Supervised` row from a
point-supervised SAM paper. In a point-supervision paper, report it explicitly
as something like `SegFormer-B0 (Full Sup., no prompt)` and compare only on
the same dataset, split, preprocessing, and evaluation protocol.

This repository-controlled baseline is designed for fair controlled comparison;
it is not a claim that the original NVLabs training recipe has been reproduced
bit-for-bit.
