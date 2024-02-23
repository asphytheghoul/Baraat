from transformers import AutoTokenizer

llama2 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
baarat = AutoTokenizer.from_pretrained("./baarat-kan-0.1")

sample_sentence = " 'Where is the poison!', exclaimed John Sparrow. ನೀವು / ನೀನು ನನ್ನಾ ಜೋಟೆ ಡಯಾನ್ಸ್ ಮಾಡ್ಟೆರಾ / ಅಯಾ?"

llama2_tokens = llama2.tokenize(sample_sentence)
baarat_tokens = baarat.tokenize(sample_sentence)
print("LLama-2 tokens: ", llama2_tokens)   
print("Baarat tokens: ", baarat_tokens)

llama2_ids = llama2.encode(sample_sentence)
baarat_ids = baarat.encode(sample_sentence)
print("LLama-2 ids: ", llama2_ids)
print("Baarat ids: ", baarat_ids)

llama2_decoded = llama2.decode(llama2_ids)
baarat_decoded = baarat.decode(baarat_ids)
print("LLama-2 decoded: ", llama2_decoded)
print("Baarat decoded: ", baarat_decoded)

print("Length of LLama-2 tokens: ", len(llama2_tokens))
print("Length of Baarat tokens: ", len(baarat_tokens))
print("Length of LLama-2 ids: ", len(llama2_ids))
print("Length of Baarat ids: ", len(baarat_ids))
