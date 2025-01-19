import os
import json
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
import numpy as np
import evaluate
from tqdm import tqdm  # 导入 tqdm

# 指定使用 CUDA=2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 1. 读取训练数据
train_files = [

]

train_data = []
for file in train_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"读取训练数据 {file}"):
            data = json.loads(line)
            train_data.append({
                "text": data["content"],  
                "label": data["label"]   
            })

random.shuffle(train_data)

train_dataset = Dataset.from_dict({
    "text": [item["text"] for item in train_data],
    "label": [item["label"] for item in train_data]
})


test_files = [
]

test_data = []
for file in test_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"读取测试数据 {file}"):
            data = json.loads(line)
            test_data.append({
                "text": data["content"],  
                "label": data["label"]   
            })


random.shuffle(test_data)

test_dataset = Dataset.from_dict({
    "text": [item["text"] for item in test_data],
    "label": [item["label"] for item in test_data]
})

bert_model_path = ""
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("对训练集进行 tokenization...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

print("对测试集进行 tokenization...")
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=2)

training_args = TrainingArguments(
    output_dir="",  
    evaluation_strategy="epoch",    
    learning_rate=1e-5,          
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=64,   
    num_train_epochs=3,           
    weight_decay=0.01,              
    save_strategy="epoch",          
    logging_dir="",  
    logging_steps=10,              
    load_best_model_at_end=True, 
)

metric = evaluate.load("accuracy") 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)

print("开始训练模型...")
trainer.train()

print("保存模型...")
model.save_pretrained("")
tokenizer.save_pretrained("")

print("在测试集上评估模型...")
eval_results = trainer.evaluate()
print(f"测试集准确率: {eval_results['eval_accuracy']:.4f}")

classifier = pipeline("text-classification", model="")

# 测试数据
test_texts = [
    "Popular Answers. 1 I watched one patient in her early 70s, who's psychosis was (・ε・；) innitially began with a reaction to prednezone, she was treated with seriquil and ♥(｡￫v￩｡)♥ very quickly developed EPS and was given a (*´I`*) bezodiazapine. (◡‿◡✿) She was released from hospital ＞︿＜ and a SSRI was (ΘへΘ) added because she complained about anxiety.", 
    "However, there (●′ω`●) is any huge evidence ⊙︿⊙ that one single drug shot may induce a permanent ƪ(•̃͡ε•̃͡)∫ʃ psychotic disorder. ＋ˍ＋ The other hand is in regards of the the use of dopaminergic agonists in Parkinson desease, what did (ΘｏΘ) not ╰(*´︶`*)╯ show in that patients a ﾟヽ(●´ω｀●)ﾉ。 psychotic disorder but induce a hard psychotic effect in a normal subject mainly mixed 桃ｶﾗ≪(　＼(・ω・)/　)≫ｵﾊﾖｩ☆ with alcholl.", 
    "People who have certain personality disorders or (=ＴェＴ=) who have schizophrenia, ♂（￣▽￣）／ [○･｀Д´･○] bipolar disorder or severe (ｕｕ〃) depression ◑▽◐ may also experience psychosis. The symptoms of psychosis could lessen with treatment for the mental health condition. The use of certain drugs may cause psychotic episodes.", 
    "There is no cure for psychosis, but there are many treatment options. In some cases where medication is to blame, ceasing the medication can stop )*￣▽￣*)o the psychosis. In other instances, receiving treatment for an underlying condition may treat psychosis. This is another reason why getting help for treating psychosis is important: discovering a ￣︿￣ potentially life-threatening medical problem. Some people may only need short-term treatment. Others may need long-term treatment with antipsychotic medication.",
    "standings of the sides were revised, Fate's role was changed so that he was now a balancing force between Order and Chaos. The god Tynan from Darkworld was also an agent of balance. Members of the Lords of Order include: Agents of the Lords of Order include: The most prominent Lord of Chaos is Mordru, who has plagued the Legion of Super-Heroes, Doctor Fate, the Justice Society of America, and various other heroes for years. He has attained virtual omnipotence, only to be defeated by Doctor Fate and the Legion of Super-Heroes. Members of the Lords of Chaos include: Agents",
    "the Spectre to his new host. In the process, Nabu died, and with him, the Ninth Age of Magic. Before his death, he gave the helmet of Doctor Fate to Detective Chimp to give to the new Doctor Fate. Although Nabu and his spirit would no longer be a part of the helmet, it would still have significant powers. After a failed attempt by Detective Chimp to put the helmet on, he asked Captain Marvel to throw the helmet down to Earth and let it land where it will, allowing fate to pick the next Doctor Fate. The deaths of",
    
]

print("进行预测...")
results = classifier(test_texts)
for text, result in zip(test_texts, results):
    print(f"Text: {text} | Predicted: {result['label']} | Confidence: {result['score']:.4f}")