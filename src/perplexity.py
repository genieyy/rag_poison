import json
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

enc = tiktoken.get_encoding("cl100k_base")

def calculate_perplexity(text):
    tokens = enc.encode(text)  
    token_probabilities = np.ones(len(tokens)) / len(tokens)
    perplexity = np.exp(-np.sum(np.log(token_probabilities)) / len(tokens))
    return perplexity

total_code_corpus_path = ''
inject_code_path = ''

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

total_code_corpus = read_jsonl(total_code_corpus_path)
inject_code = read_jsonl(inject_code_path)

perplexities = [calculate_perplexity(item['content']) for item in tqdm(total_code_corpus, desc="Calculating Perplexities")]

inject_perplexities = [calculate_perplexity(item['content']) for item in inject_code]

plt.figure(figsize=(15, 9))
violin_parts = plt.violinplot(perplexities, vert=False, widths=0.7, showmeans=False, showmedians=True, bw_method=0.1)

if 'cmedians' in violin_parts:
    violin_parts['cmedians'].set_color('blue')
    violin_parts['cmedians'].set_linewidth(5)

for spine in plt.gca().spines.values():
    spine.set_linewidth(3)  
    spine.set_color('black')  



plt.scatter(inject_perplexities, np.ones(len(inject_perplexities)), color='red', label='Malicious Data', zorder=30, s=200)

plt.title('NQ dataset', fontsize=45)
plt.xlabel('Perplexity', fontsize=45)
plt.ylabel('Density', fontsize=45)
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)

plt.xlim(100,250)

plt.plot([], [], color='blue', label='Data distribution median')

legend = plt.legend(
    loc='upper right', 
    fontsize=45,        
    markerscale=2,     
    borderpad=1,       
    handlelength=2,    
    handletextpad=1,   
)

for line in legend.get_lines():
    line.set_linewidth(3) 

plt.grid(True, linestyle='--', alpha=0.7, linewidth= '3')
plt.tight_layout()

plt.savefig("", format='svg', dpi=50, transparent=False)
plt.savefig('')
plt.show()