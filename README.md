# Mamba Dive 🐍

This is the code that goes along with our [Practical ML Dive](https://lu.ma/practicalml) into Mamba. The full experiment details and notes are captured in this blog post:

[https://blog.oxen.ai/practical-ml-dive-how-to-train-mamba-for-question-answering/](https://blog.oxen.ai/practical-ml-dive-how-to-train-mamba-for-question-answering/)

~ TLDR ~

The 2.8b model shows a little promise for Question Answering tasks on SQuAD with a little prompt engineering, but overall the model felt a little underwhelming. It got `7.5%` accuracy on a held out set of the [SQuAD eval set](https://www.oxen.ai/ox/Mamba-Fine-Tune/file/main/squad_val_1k.jsonl).

The 130m model did a little better than prompt engineering the 2.8b after a full train, but still only got `12.5%` accuracy with a raw string match on the held out.

The models are trained to be able to say "I don't know" if there is no context that supports answering the question which I feel is an important property of a QA model.

The trained 130m model can be found on [hugging face](https://huggingface.co/Oxen-AI/mamba-130m-context).

There is code to train models of any size on the SQuAD dataset. The dataloader puts all the data into context-question-answer triples like so

```
{context}

Q: {question}
A: {answer}
```

For example:

```
The Panthers used the San Jose State practice facility and stayed at the San Jose Marriott.
The Broncos practiced at Stanford University and stayed at the Santa Clara Marriott.

Q: What hotel did the Panther’s stay at?
A: San Jose Marriott
```

There are scripts to evaluate the models on larger datasets as well as run on the CLI. All the data can be found on Oxen.ai

[https://www.oxen.ai/ox/Mamba-Fine-Tune](https://www.oxen.ai/ox/Mamba-Fine-Tune)

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