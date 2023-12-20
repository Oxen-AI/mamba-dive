import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import sys

# Validate CLI params
if len(sys.argv) != 2:
    print("Usage: python train.py state-spaces/mamba-130m")
    exit()

# Take in the model you want to train
model_name = sys.argv[1]

# Choose a tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

# Instantiate the MambaLMHeadModel from the state-spaces/mamba GitHub repo
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173
model = MambaLMHeadModel.from_pretrained(model_name, device="cuda", dtype=torch.float16)

while True:
    # Take the user input from the command line
    user_message = input("\n> ")

    # Create a prompt
    n_shot_prompting = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "question": "Who invented the segway?",
            "answer": "Dean Kamen"
        },
        {
            "question": "What is the fastest animal?",
            "answer": "Cheetah"
        }
    ]

    prompt = f"You are a Trivia QA bot.\nAnswer the following question succinctly and accurately."
    prompt = f"{prompt}\n\n" + "\n\n".join([f"Q: {p['question']}\nA: {p['answer']}" for p in n_shot_prompting])
    prompt = f"{prompt}\n\nQ: {user_message}"

    # Debug print to make sure our prompt looks good
    print(prompt)

    # Encode the prompt into integers and convert to a tensor on the GPU
    input_ids = torch.LongTensor([tokenizer.encode(prompt)]).cuda()
    
    # Generate an output sequence of tokens given the input
    # "out" will contain the raw token ids as integers
    out = model.generate(
        input_ids=input_ids,
        max_length=128,
        eos_token_id=tokenizer.eos_token_id
    )

    # you must use the tokenizer to decode them back into strings
    decoded = tokenizer.batch_decode(out)[0]
    print("="*80)

    # out returns the whole sequence plus the original
    cleaned = decoded.replace(prompt, "")

    # the model will just keep generating, so only grab the first one
    cleaned = cleaned.split("\n\n")[0]
    print(cleaned)