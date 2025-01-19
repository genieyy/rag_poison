import json
import subprocess
from tqdm import tqdm
import time
import signal

# Set your API key
api_key = ''

input_file = ''

total_count = 0
correct_count = 0
correct_ids = []

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def gpt4o_judge(answer, groundtruth, max_retries=10, retry_delay=2, timeout=5):

    prompt = f"Determine whether the following answers are correct:\n\nThe correct answer to the question is: '{groundtruth}'\n\nThe model's answer is: '{answer}'\n\nIf the model's answer is correct, please answer 'yes' first; otherwise, please answer 'no' first."
    
    for attempt in range(max_retries):
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            curl_command = [
                "curl", "https://api.chatanywhere.tech/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", json.dumps({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0
                })
            ]
            result = subprocess.run(curl_command, capture_output=True, text=True)
            signal.alarm(0)
            response = json.loads(result.stdout)
            reply = response['choices'][0]['message']['content'].strip()
            if 'yes' in reply:
                return True
            else:
                return False
        except TimeoutException:
            print(f"Timeout occurred, retrying... ({attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"Error calling GPT-4 API: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                return False

def compute_similarity(answer, groundtruth):
    """
    Compute the proportion of overlapping words between the answer and groundtruth.
    Returns a similarity score between 0 and 1.
    """
    answer_tokens = set(answer.lower().split())
    groundtruth_tokens = set(groundtruth.lower().split())
    overlap = answer_tokens & groundtruth_tokens
    if len(groundtruth_tokens) == 0:
        return 0
    similarity = len(overlap) / len(groundtruth_tokens)
    return similarity

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = lines[:-1]

for line in tqdm(lines, desc="Processing lines"):
    data = json.loads(line)
    answer = data.get('answer', '').lower()
    groundtruth = data.get('groundtruth', '').lower()
    total_count += 1

    if ("sorry" not in answer ):
        
        is_correct_by_gpt4o = gpt4o_judge(answer, groundtruth)
        time.sleep(0.5) 

        if is_correct_by_gpt4o:
            correct_count += 1
            correct_ids.append(data.get('id'))

error_rate = (total_count - correct_count) / total_count * 100

print(f"总共有 {total_count} 条数据，正确答案的数量为 {correct_count} 条，错误率为 {error_rate:.6f}%")

print("回答正确的条目如下：")
for idx, id in enumerate(correct_ids, start=1):
    print(f"ID: {id}", end="\t") 
    if idx % 5 == 0:  
        print() 
if len(correct_ids) % 5 != 0:
    print()