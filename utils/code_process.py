from human_eval.data import write_jsonl, stream_jsonl
import glob 
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument(
    '--path',
    type=str,
    help="")
parser.add_argument(
    '--out_path',
    type=str,
    help="")

args = parser.parse_args()
humaneval_output = []
mbpp_output = []
for code in stream_jsonl(args.path):
    if code['type'] not in ['humaneval', 'mbpp']:
        continue
    task_id = code['answer']
    code['task_id'] = str(task_id)
    completion = code['output'].replace("\r", "")            
    if '```python' in completion: 
        def_line = completion.index('```python')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('\n```')
            completion = completion[:next_line].strip()
        except:
            pass
    
    if "__name__ == \"__main__\"" in completion:
        next_line = completion.index('if __name__ == "__main__":')
        completion = completion[:next_line].strip()
    
    if "# Example usage" in completion:
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()
    
    if "assert" in completion:
        next_line = completion.index('assert')
        completion = completion[:next_line].strip()
    
    code['completion'] = completion
    
    if code['type'] == 'humaneval':
        humaneval_output.append(code) 
    else:
        mbpp_output.append(code)
    
import os
humaneval_outpath = os.path.join(os.path.dirname(args.path), "humaneval.jsonl")
mbpp_outpath = os.path.join(os.path.dirname(args.path), "mbpp.jsonl")

print("save to {}".format(humaneval_outpath))
print("save to {}".format(mbpp_outpath))
write_jsonl(humaneval_outpath, humaneval_output)
write_jsonl(mbpp_outpath, mbpp_output)