import argparse
import re
from fraction import Fraction
import json

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help="")
args = parser.parse_args()

result = []
with open(args.input_file, 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        y_pred = extract_answer_number(data['output'])
        if y_pred != None:
            result.append(float(y_pred) == float(data["answer"]))
        else:
            result.append(False)
acc = sum(result) / len(result)
print('gsm8k length====', len(result), ', gsm8k acc====', acc)