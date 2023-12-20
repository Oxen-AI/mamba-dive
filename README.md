# Mamba Dive 🐍

This is the code that goes along with our [Practical ML Dive](https://lu.ma/practicalml) into Mamba

# N-Shot Prompt Engineering

```bash
python prompt_mamba.py state-spaces/mamba-130m
```

# Eval Data

[https://www.oxen.ai/ox/Mamba-Fine-Tune](https://www.oxen.ai/ox/Mamba-Fine-Tune)

```bash
oxen clone https://hub.oxen.ai/ox/Mamba-Fine-Tune
```

# Eval N-Shot

```bash
python eval_n_shot_mamba.py state-spaces/mamba-130m data/Mamba-Fine-Tune/squad_val_1k.jsonl data/Mamba-Fine-Tune/results/results.jsonl
oxen df data/Mamba-Fine-Tune/results/results.jsonl --sql "SELECT is_correct, COUNT(*) FROM df GROUP BY is_correct;"
```

# Train Context Aware QA

```bash
python train_mamba_with_context.py --model state-spaces/mamba-130m \
   --data_path data/Mamba-Fine-Tune/squad_train.jsonl \
   --output models/mamba-130m-context \
   --num_epochs 10
```

# Context Aware QA

```bash
python prompt_mamba_with_context.py models/mamba-130m-context
```

# Eval Context Aware QA

```bash
python eval_mamba_with_context.py models/mamba-130m-context data/Mamba-Fine-Tune/squad_val_1k.jsonl data/Mamba-Fine-Tune/results_context.jsonl
```