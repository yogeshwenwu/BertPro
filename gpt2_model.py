# Import necessary packages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load the Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
device = "cpu"  # device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
max_new_token = 10

# Input prompt as batch
sentences = [
    "The weather is bad today it might rain anytime",
    "Artificial intelligence is transforming the way ",
    "The movie I watched yesterday had an unexpected twist at the end ",
    "you recommended a good book to read over the weekend, that was",
    "The capital of France is Paris, known for its art, culture ",
    "She ordered a latte at the café and worked on her presentation ",
    "the key differences between machine learning and deep learning is ",
    "The traffic on my way to work this morning was ",
    "Python is a versatile programming language often used in ",
    "He went to the gym every day, determined to improve"
]



num_sentences = len(sentences)

# tokenization and  it's time
enc_start = time.time()
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
enc_time = time.time() - enc_start

input_token = encoded_input["input_ids"].numel()
# # warmup run
# with torch.no_grad():
#     _ = model.generate(
#     **encoded_input,
#     max_new_tokens=max_new_token,
#     num_beams=4,
#     return_dict_in_generate=True,
#     output_scores=True,
# )

ttft_start = time.time()
with torch.no_grad():
    outputs = model.generate(
    **encoded_input,
    max_new_tokens=1,
    num_beams=4,
    return_dict_in_generate=True,
    output_scores=True,
)
ttft_end = time.time()

start_time = time.time()
with torch.no_grad():
    outputs = model.generate(
    **encoded_input,
    max_new_tokens=max_new_token,
    num_beams=4,
    do_sample = True,
    temperature = 0.7,
    repetition_penalty=1.2,
    return_dict_in_generate=True,
    output_scores=True,
)
end_time = time.time()

decode_start = time.time()
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
decode_time = time.time() - decode_start
# print(outputs)
num_tokens_generated = sum(len(output) for output in outputs.sequences)

ttft_time = (ttft_end - ttft_start)
tpot_time = ((end_time - start_time) - ttft_time) / (num_tokens_generated - 1)

total_time = enc_time + (ttft_time + tpot_time) +decode_time

# need to remove input tokens from output tokens
num_tokens_generated -= input_token

print(f"Number of Tokens generated:{num_tokens_generated}") # 210
print(f"Number of sentences: {num_sentences}") # 10

latency_perBatch = total_time / num_sentences # per-batch = (tot_time/ num_itr)
latency = total_time # entire batch = tot_time
tps = num_tokens_generated / (latency / 1000) 
rps = num_sentences / (latency / 1000)

# cross verify latency, ttft, tpot using the below formula
crossVerify_latency = ttft_time + (tpot_time * (max_new_token - 1))
if (crossVerify_latency == latency - enc_time - decode_time):
    print("Correct Latency") # if correct 
else:
    print("Incorrect Latency")
print()


# print Preformance measures    
# print(f"Input tokens: {input_token}")
print(f"CVLatency: {crossVerify_latency * 1000:.4f} ms")
print(f"Latency: {latency * 1000:.4f} ms")
print(f"TTFT: {ttft_time:.4f} s")
print(f"TPOT: {tpot_time:.4f} tps")
print(f"TPS: {tps:.4f} tps")
print(f"RPS: {rps:.4f} rps")


