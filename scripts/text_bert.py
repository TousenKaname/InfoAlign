from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
text = "bert-base-uncasedbert-base-uncasedbert-base-uncasedbert-base-uncasedbert-base-uncased"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output.last_hidden_state.shape)