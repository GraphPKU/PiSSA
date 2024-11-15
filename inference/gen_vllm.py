import argparse
import torch
import sys
import os
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="")
parser.add_argument('--output_file', type=str, default="gsm8k_response.jsonl", help="")
parser.add_argument("--data_path", type=str, default="inference/data/eval_gsm8k")
parser.add_argument("--batch_size", type=int, default=400, help="")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=1024, help="")
args = parser.parse_args()

stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop=stop_tokens)
llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count())

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data

dataset = load_from_disk(args.data_path)
batch_dataset_query = batch_data(dataset["query"], batch_size=args.batch_size)
batch_dataset_answer = batch_data(dataset["answer"], batch_size=args.batch_size)

for idx, (batch_query, batch_answer) in enumerate(zip(batch_dataset_query, batch_dataset_answer)):
    with torch.no_grad():
        completions = llm.generate(batch_query, sampling_params)
    for query, completion, answer in zip(batch_query, completions, batch_answer):
        with open(os.path.join(args.model, args.output_file), 'a') as f:
            json.dump({'question': query, 'output': completion.outputs[0].text, 'answer': answer}, f)
            f.write('\n')
