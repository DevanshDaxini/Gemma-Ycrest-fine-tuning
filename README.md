# Gemma 3 1B LoRA Fine-tuning Practice (MLX)

Two-phase fine-tuning project on Apple Silicon (M4, 16GB).  
**Phase 1**: nonsense word mapping — pipeline validation.  
**Phase 2**: workout log → NDJSON report — structured generation skill.  
Phase 2 is a skill-transfer prototype; its schema is a placeholder for a downstream NDJSON revenue-brief project.

---

## One-time Setup

```bash
pip install -r requirements.txt

# Gemma 3 requires HuggingFace login (gated model)
huggingface-cli login
# Accept the Gemma license at: https://huggingface.co/google/gemma-3-1b-it
```

### Model notes
Both `lora_config.yaml` files default to `mlx-community/gemma-3-1b-it-4bit`.  
If that repo doesn't exist yet, change `model:` to `google/gemma-3-1b-it`; mlx_lm converts on first use.

---

## Phase 1: Nonsense Word Mapping

### Run order

```bash
cd phase1_toy

# 1. Generate data
python generate_data.py
# → data/train.jsonl (~315 ex), data/valid.jsonl (~35 ex), data/test.jsonl (~105 ex)

# 2. Train
bash train.sh
# → adapters/ saved every 100 steps

# 3. Inference (single prompt)
python infer.py --prompt "Map: <word>" --mode greedy
python infer.py --prompt "Map: <word1> <word2>" --mode distribution --top-k 5

# 4. Evaluate
python eval.py
```

### Expected runtime (M4, 16GB)
| Step | Time |
|------|------|
| generate_data.py | < 5 s |
| train.sh (250 iters) | ~5–8 min |
| eval.py (105 examples) | ~8–12 min |

### Expected accuracy
| Category | Expected range |
|----------|----------------|
| Singleton (memorisation) | 90–100% |
| Seen composition | 80–95% |
| **Held-out composition** | **60–85%** ← key metric |
| Triple (stretch) | 40–70% |

---

## Phase 2: Workout Log → NDJSON

### Run order

```bash
cd phase2_ndjson

# 1. Generate data
python generate_data.py
# → data/train.jsonl (150 ex), data/valid.jsonl (15 ex), data/test.jsonl (40 ex)

# 2. Train
bash train.sh
# → adapters/ (120 iters ≈ 3 epochs)

# 3. Inference
python infer.py --prompt "Alex 45min BP:3x8@80kg SQ:4x5@100kg" --mode greedy
# → pretty-prints NDJSON, flags invalid lines

# 4. Evaluate (tiered)
python eval.py
```

### Expected runtime (M4, 16GB)
| Step | Time |
|------|------|
| generate_data.py | < 5 s |
| train.sh (120 iters) | ~3–5 min |
| eval.py (40 examples) | ~6–10 min |

### Expected tiered accuracy
| Tier | Expected range |
|------|----------------|
| NDJSON validity | 85–95% |
| Action ID validity | 80–92% |
| Schema adherence | 75–88% |
| Semantic accuracy | 60–80% |

---

## How to Swap Adapters

Each phase saves adapters to `./adapters/`. To use a specific checkpoint:

```bash
# Use a specific adapter step
python infer.py --adapter ./adapters/  # uses latest

# Compare base vs fine-tuned
python infer.py --prompt "..." --no-adapter      # base model
python infer.py --prompt "..." --adapter ./adapters/  # fine-tuned

# Swap Phase 1 adapter into Phase 2 infer (for cross-phase experiments)
python phase2_ndjson/infer.py --adapter ./phase1_toy/adapters/
```

The adapter directory must contain `adapter_config.json` + `adapters.npz`.

---

## Reading the Loss Curve

Training logs look like:

```
Iter 10: Train loss 2.843, It/sec 1.21, Tokens/sec 456.3
Iter 25: Val loss 2.601
Iter 50: Train loss 1.924 ...
```

- **Train loss falling quickly**: model memorising training examples.
- **Val loss tracking closely**: good generalisation, no overfitting.
- **Val loss rising while train loss falls**: overfitting — reduce iters or lower LR.
- Typical healthy loss at convergence: 0.3–0.8 for Phase 1, 0.8–1.5 for Phase 2.

To plot the loss (save stdout to a file first):
```bash
bash train.sh 2>&1 | tee train_log.txt
grep "Train loss" train_log.txt | awk '{print NR, $5}' > loss.txt
```

---

## Troubleshooting (M4, 16GB)

### OOM / killed during training
```bash
# Option 1: reduce batch size in lora_config.yaml
batch_size: 2

# Option 2: reduce max_seq_length
max_seq_length: 128   # Phase 1
max_seq_length: 256   # Phase 2

# Option 3: use 4-bit base model (default) and verify
model: "mlx-community/gemma-3-1b-it-4bit"
```

### "model not found" error
```bash
# mlx-community mirror may not exist yet; use HF directly
model: "google/gemma-3-1b-it"
# mlx_lm auto-converts safetensors → mlx on first load (~1 min)
```

### Very slow generation in infer.py (distribution mode)
Distribution mode re-runs the full forward pass per token step (no KV cache).  
Normal for 10 steps. Use `--n-tokens 5` to halve it.

### Adapter load error: "adapter_config.json not found"
Run training first (`bash train.sh`) before running infer/eval.

### Loss stuck at high value after 50 iters
Try a warmup run with a higher LR first, or increase `lora_layers` from 16 → 18 in the yaml.

### huggingface-cli: token not found
```bash
huggingface-cli login --token hf_YOUR_TOKEN
```

---

## Project Structure

```
gemma-finetune-practice/
├── phase1_toy/
│   ├── generate_data.py   word-mapping dataset generator
│   ├── train.sh           invokes mlx_lm.lora
│   ├── infer.py           greedy + distribution inference
│   ├── eval.py            4-category accuracy report
│   ├── lora_config.yaml   training hyperparameters
│   └── data/              train / valid / test JSONL
├── phase2_ndjson/
│   ├── schema.py          Pydantic schemas + 18 action IDs
│   ├── generate_data.py   synthetic workout log generator
│   ├── train.sh
│   ├── infer.py           NDJSON pretty-print + validation
│   ├── eval.py            tiered evaluation + confusion matrix
│   ├── lora_config.yaml
│   └── data/
├── shared/
│   └── mlx_utils.py       Gemma template, load, greedy, distribution helpers
├── requirements.txt
└── README.md
```
