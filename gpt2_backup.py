# Import necessary packages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load the Tokenizer & Model

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

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

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
device = "cpu"  # device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}


ttft_start = time.time()
with torch.no_grad():
    _ = model.generate(
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
    max_new_tokens=10,
    num_beams=4,
    do_sample = True,
    temperature = 0.7,
    repetition_penalty=1.2,
    return_dict_in_generate=True,
    output_scores=True,
)
end_time = time.time()

print(outputs)

generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
num_tokens_generated = sum(len(output) for output in outputs.sequences)


total_time = (end_time - start_time) * 1000

latency = total_time / num_sentences
ttft_time = (ttft_end - ttft_start)
tpot_time = ((total_time - ttft_time) / 1000) / num_tokens_generated 
tps = num_tokens_generated / (latency / 1000)
rps = num_sentences / latency
print(f"Latency: {latency:.4f} ms")
print(f"TTFT: {ttft_time:.4f} s")
print(f"TPOT: {tpot_time:.4f} tps")
print(f"TPS: {tps:.4f} tps")
print(f"RPS: {tps:.4f} rps")


