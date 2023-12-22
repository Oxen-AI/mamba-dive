import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
import pandas as pd
import time
from tqdm import tqdm

def run_mistral(model, tokenizer, question, context):

    text = f"You are a Trivia QA bot. Answer the following trivia question given the context above. Answer the question with a single word if possible. For example:\n"
    
    n_shot_prompting = [
        {
            "context": "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km2 (41 sq mi), Paris is the fifth-most populated city in the European Union and the 30th most densely populated city in the world in 2022.",
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "context": "Dean Lawrence Kamen (born April 5, 1951) is an American engineer, inventor, and businessman. He is known for his invention of the Segway and iBOT,[2] as well as founding the non-profit organization FIRST with Woodie Flowers. Kamen holds over 1,000 patents.",
            "question": "Who invented the segway?",
            "answer": "Dean Kamen"
        },
        {
            "context": "The peregrine falcon is the fastest bird, and the fastest member of the animal kingdom, with a diving speed of over 300 km/h (190 mph). The fastest land animal is the cheetah. Among the fastest animals in the sea is the black marlin, with uncertain and conflicting reports of recorded speeds.",
            "question": "What is the fastest animal?",
            "answer": "Cheetah"
        }
    ]

    text = f"{text}\n\n" + "\n\n".join([f"Context: {p['context']}\n\nQuestion: {p['question']}\n{p['answer']}" for p in n_shot_prompting])
    text = f"{text}\n\nContext: {context}\n\nQuestion: {question}\n"
    # print(text)
    
    messages = [
        {"role": "user", "content": text}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    
    # print(text)
    # input_ids = torch.LongTensor([tokenizer.encode(text)]).cuda()
    input_ids = encodeds.cuda()
    # print(input_ids)
    
    out = model.generate(
        input_ids,
        max_new_tokens=128,
        do_sample=True
    )

    # print(out)
    decoded = tokenizer.batch_decode(out)[0]
    # print("="*80)
    # print(decoded)
    
    # out returns the whole sequence plus the original
    cleaned = decoded.split("[/INST]")[-1]
    cleaned = cleaned.replace("</s>", "")
    
    # # the model will just keep generating, so only grab the first one
    answer = cleaned.split("\n\n")[0].strip()
    # answer = cleaned.strip()
    # print(answer)
    return answer, input_ids.shape[1]

def write_results(results, output_file):
    df = pd.DataFrame(results)
    df = df[["idx", "context", "question", "answer", "guess", "is_correct", "time", "num_tokens", "tokens_per_sec"]]

    print(f"Writing {output_file}")
    df.to_json(output_file, orient="records", lines=True)

model_name = sys.argv[1]
dataset = sys.argv[2]
output_file = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map = "auto").cuda()

results = []
with open(dataset) as f:
    all_data = []
    for line in tqdm(f):
        data = json.loads(line)
        all_data.append(data)

    total_qs = len(all_data)
    for i, data in enumerate(all_data):
        start_time = time.time()

        # print(data)
        question = data["prompt"]
        context = data["context"]
        answer = data["response"]
        guess, num_tokens = run_mistral(model, tokenizer, question, context)
        end_time = time.time()
        is_correct = (answer.strip().lower() in guess.strip().lower())
        print(f"Question {i}/{total_qs}")
        print(f"Context: {context}")
        print(f"Num Tokens: {num_tokens}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"?: {guess}")
        if is_correct:
            print(f"✅")
        else:
            print(f"❌")
        print("="*80)
        total_time = end_time - start_time
        result = {
            "idx": i,
            "question": question,
            "context": context,
            "answer": answer,
            "guess": guess,
            "is_correct": is_correct,
            "time": total_time,
            "num_tokens": num_tokens,
            "tokens_per_sec": (num_tokens/total_time)
        }
        results.append(result)

        if len(results) % 20 == 0:
            write_results(results, output_file)
            
        # if len(results) > 100:
        #     break

write_results(results, output_file)
