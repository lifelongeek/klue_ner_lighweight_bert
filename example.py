from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

#dataset = load_dataset('klue') # not work why?

model = AutoModel.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

sent = "엔씨소프트는 블소2를 올 상반기 선보일 예정이었으나 넷마블 '제2의 나라', 카카오게임즈 '오딘' 등 경쟁사 게임 출시가 몰리면서 출시시점을 올 3분기로 미뤘다."
tokens = tokenizer(sent, return_tensors='pt')

input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
output = model(input_ids=input_ids, attention_mask=attention_mask)
