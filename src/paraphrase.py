import openai
import json

openai.api_key = ""
def generate_paraphrases(question):
    prompt = f"This is my question: {question}. Please craft 5 paraphrased versions for the question. Please do not change the meaning of the original question! Give your reply as a JSON formatted string. The reply should use “paraphrased_questions” as key, [question1, question2, question3, question4, question5] as value."
    response = openai.ChatCompletion.create(
        model="gpt-4",   
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    return response.choices[0].message['content'].strip()

def read_questions_from_jsonl(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data)
    return questions

def write_paraphrased_questions_to_jsonl(questions, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for question_data in questions:
            original_id = question_data['id']
            original_question = question_data['question']
            groundtruth = question_data['groundtruth']
            
            paraphrased_questions_json = generate_paraphrases(original_question)
            paraphrased_questions = json.loads(paraphrased_questions_json)['paraphrased_questions']
            
            for i, paraphrased_question in enumerate(paraphrased_questions, start=1):
                new_id = f"{original_id}_{i}"
                new_question_data = {
                    "id": new_id,
                    "question": paraphrased_question,
                    "groundtruth": groundtruth
                }
                file.write(json.dumps(new_question_data, ensure_ascii=False) + '\n')

input_file_path = ""
output_file_path = ""

questions = read_questions_from_jsonl(input_file_path)

write_paraphrased_questions_to_jsonl(questions, output_file_path)

print(f"Paraphrased questions have been written to {output_file_path}")