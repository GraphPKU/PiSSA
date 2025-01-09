import argparse
import torch
import sys
import os
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="")
parser.add_argument("--data_path", type=str, default="pissa-dataset")
parser.add_argument('--sub_task', nargs='+', help='')
parser.add_argument('--dataset_split', type=str, default="test", help='')
parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="")
parser.add_argument("--batch_size", type=int, default=400, help="")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=1024, help="")
args = parser.parse_args()

stop_tokens = []
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

if args.sub_task is None:
    dataset = load_dataset(args.data_path, split=args.dataset_split)
else:
    all_test_dataset = []
    for task in args.sub_task:
        ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
        print(f"{args.data_path}/{task}/{args.dataset_split}")
        for k,v in ds[0].items():
            print("-"*100)
            print(k,end=':\t')
            print(v)
        print("+"*100)
        all_test_dataset.append(ds)
        
    dataset = concatenate_datasets(all_test_dataset)
    
batch_dataset_query = batch_data(dataset["instruction"], batch_size=args.batch_size)
batch_dataset_answer = batch_data(dataset["output"], batch_size=args.batch_size)
batch_dataset_task = batch_data(dataset["type"], batch_size=args.batch_size)

for idx, (batch_query, batch_answer, batch_task) in enumerate(zip(batch_dataset_query, batch_dataset_answer,batch_dataset_task)):
    with torch.no_grad():
        completions = llm.generate(batch_query, sampling_params)
    for query, completion, answer, task in zip(batch_query, completions, batch_answer, batch_task):
        with open(args.output_file, 'a') as f:
            json.dump({'type': task, 'query': query, 'output': completion.outputs[0].text, 'answer': answer}, f)
            f.write('\n')
