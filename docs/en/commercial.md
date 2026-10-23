# Commercial route

This project is for non-commercial use. Open-source weights, including ones published for non-commercial research, may be downloaded by the operator. The registry still marks every built-in row `commercial_use: false`, and the files are not shipped in this tree.

`--commercial` sets `execution.commercial`. The gate refuses every role the run would open when that registry row has `commercial_use: false` or is missing. Every built-in model is marked false, so a commercial build cannot load inswapper, the optional swappers, GFPGAN, CodeFormer, GPEN, RestoreFormer++, the occluder, or the parser. There is no config key that flips those rows.

A commercially licensable swapper is something you train and register yourself. The scaffold follows the idea of an identity-conditioned generator in the Apache-2.0 GHOST / AEI-Net family. It does not copy their training source, and it does not download their weights.

```bash
python scripts/train_swapper.py --dataset configs/dataset.example.yaml --check
python scripts/export_onnx.py --check
python scripts/eval_swapper.py --self-test
```

`configs/dataset.example.yaml` is a manifest. `license` must be the string `commercial-ok` on the set and on every sample. Resolution is 256 or 512. The checker does not open the image paths and does not fetch them.

Without `--check`, `train_swapper.py` prints the plan and exits 2. `export_onnx.py` prints the contract (NCHW RGB normalized to [-1, 1], a 512-d latent, ArcFace template scaled by size/112) and exits 2 when no checkpoint is given. Evaluation is global SSIM plus cosine similarity of two embeddings the caller already has.

After legal review, add a registry row with `commercial_use: true`. The gate reads that flag and nothing else.
