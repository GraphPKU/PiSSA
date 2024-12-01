import argparse
import re
import json

def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_interaction_qa', 'arc_challenge', 'arc_easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help="")
parser.add_argument('--input_file', type=str, help="")
args = parser.parse_args()

result = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        y_pred = extract_answer(args.dataset ,data['output'])
        if y_pred != None:
            result.append(y_pred == data["answer"])
        else:
            result.append(False)
acc = sum(result) / len(result)
print('gsm8k length====', len(result), ', gsm8k acc====', acc)