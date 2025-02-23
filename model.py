#Doubts
# line no: 10, 15, 39

# Import necessary packages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# List of input sentences (batch)
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

# Tokenize the input sentences
inputs = tokenizer(sentences, padding=True, return_tensors="pt")
# inputs = tokenizer(sentences, padding=True, return_tensors="pt")

# Move to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

start_time = time.time()

# # Batch inference
with torch.no_grad():
    ttft_start = time.time()
    
#     outputs = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         top_k=50,
#         repetition_penalty=1.2 
#     )

# Batch inference when do_sample = True
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"], 
        max_length=inputs["input_ids"].shape[1] + 20, 
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2, 
        pad_token_id=tokenizer.eos_token_id 
    )
    ttft_end = time.time()
    print((ttft_end - ttft_start)*1000)
end_time = time.time()

# print(f"Outputs: {outputs.shape}")
# Decode the outputs
completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Print results
for sentence, completion in zip(sentences, completions):
    print(f"Input: {sentence}\nCompleted Sentence: {completion}\n")

total_time = end_time - start_time  
num_sentences = len(sentences)
num_tokens_generated = sum(len(output) for output in outputs)

latency = total_time / num_sentences  * 1000
throughput = num_sentences / total_time  
ttft = (ttft_end - ttft_start) * 1000
tpot = (total_time - ttft) * 1000 / num_tokens_generated  

# **Display Performance Metrics**
print(f"\nPerformance Metrics:")
print(f"Latency: {latency:.4f} ms")
print(f"Throughput: {throughput:.4f} requests per second")
print(f"TTFT: {ttft:.4f} ms")
print(f"TPOT: {tpot:.4f} ms per token")