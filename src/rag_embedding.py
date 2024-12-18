import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 设置环境变量，指定使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和分词器
model_name_or_path = '/data3/zhouxinyun/.cache/huggingface/hub/models--Salesforce--SFR-Embedding-2_R'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path).to(device)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU 进行并行计算")
    model = torch.nn.DataParallel(model)

# 定义文件路径
text_file = '/data3/zhouxinyun/rag_poison/MedRAG/src/corpus/code/total_code_corpus_rank0.jsonl'
embedding_file = '/data3/zhouxinyun/rag_poison/MedRAG/src/corpus/data/embedding_base/code/sfr/sfr_embedding_total_rank0.pt'
texts_file = '/data3/zhouxinyun/rag_poison/MedRAG/src/corpus/data/embedding_base/code/sfr/sfr_texts_total_rank0.npy'

query_file = '/data3/zhouxinyun/rag_poison/MedRAG/src/corpus/meddomain/total_med_poison_question.jsonl'
output_file = '/data3/zhouxinyun/rag_poison/MedRAG/src/corpus/data/result/meddomain_Qwen2-Embedding.jsonl'

# 加载并向量化文本内容
def load_and_embed_texts(text_file):
    # 如果已经计算过嵌入并保存过，直接加载
    if os.path.exists(embedding_file) and os.path.exists(texts_file):
        embeddings = torch.load(embedding_file, map_location=device)
        texts = np.load(texts_file, allow_pickle=True).tolist()
        print("已加载保存的嵌入和文本。")
    else:
        # 否则，计算嵌入并保存
        texts = []
        embeddings = []
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="正在向量化文本内容"):
                try:
                    item = json.loads(line)
                    text = item['content']
                    texts.append(text)
                    # 将文本转为模型输入格式
                    inputs = tokenizer(
                        text, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True, 
                        max_length=8192
                    ).to(device)  # 确保 inputs 在同一个设备上
                    # 确保 input_ids 和 attention_mask 是 Long 类型
                    inputs['input_ids'] = inputs['input_ids'].long()
                    inputs['attention_mask'] = inputs['attention_mask'].long()
                    
                    # 禁用梯度计算以加速推理
                    with torch.no_grad():
                        # 获取模型输出
                        outputs = model(**inputs)
                        # 获取模型的最后一个隐藏状态并取平均值作为嵌入
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu()
                    embeddings.append(embedding)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e} in line: {line}")
                    continue

        # 将嵌入和文本合并并保存
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, embedding_file)
        np.save(texts_file, np.array(texts))
        print("已计算并保存嵌入和文本。")
    
    return texts, embeddings

# 主函数
def main():
    # 加载并嵌入文本
    texts, embeddings = load_and_embed_texts(text_file)

if __name__ == '__main__':
    main()