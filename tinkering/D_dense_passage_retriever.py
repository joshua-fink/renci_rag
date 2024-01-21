from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")
print(tokenizer.is_fast)
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
questions = ["Hello, is my dog cute ?"]
input_ids = tokenizer(questions, return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output

print(embeddings)