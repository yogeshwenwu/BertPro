# Import necessary packages
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import time

# loading model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = None
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    quantization_config=quantization_config)
device = "cpu"
quantized_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

sentences = [
    "The weather is bad today it might rain anytime",
    "Artificial intelligence is transforming the way",
    "The movie I watched yesterday had an unexpected twist at the end",
    "You recommended a good book to read over the weekend, that was",
    "The capital of France is Paris, known for its art, culture",
    "She ordered a latte at the café and worked on her presentation",
    "The key differences between machine learning and deep learning is",
    "The traffic on my way to work this morning was",
    "Python is a versatile programming language often used in",
    "He went to the gym every day, determined to improve",
    "Quantum computing has the potential to revolutionize data encryption",
    "The latest advancements in robotics have enabled autonomous medical procedures",
    "NASA's new telescope can capture images of distant galaxies with unprecedented clarity",
    "The discovery of a new exoplanet raises questions about extraterrestrial life",
    "Meditation has been shown to improve mental clarity and reduce stress",
    "Scientists recently found a way to generate renewable energy from ocean waves",
    "The stock market saw a major shift after the latest tech industry boom",
    "Investing in cryptocurrency can be both rewarding and risky",
    "Cultural diversity enriches society by introducing new perspectives and traditions",
    "The human brain remains one of the most complex and least understood organs",
    "The ethical implications of genetic cloning continue to be widely debated",
    "If time travel were possible, would we change the past or the future",
    "The Renaissance period was a time of great artistic and intellectual growth",
    "She had a dream about a hidden city beneath the ocean waves",
    "The detective knew the case wasn’t as simple as it seemed",
    "He found an ancient map hidden inside his grandfather’s journal",
    "A simple act of kindness can brighten someone’s entire day",
    "Financial literacy is an essential skill that should be taught in schools",
    "The invention of the printing press revolutionized the spread of knowledge",
    "The old man stared at the letter, unsure if he should open it",
    "The moon shone brighter than ever before, casting an eerie glow over the city",
    "A mysterious book appeared on her doorstep with no sender address",
    "Regular exercise not only improves physical health but also boosts mood",
    "A secret tunnel beneath the library led to a forgotten underground world",
    "The radio suddenly started playing a song from the future",
    "The cat stared at the empty space, as if it could see something invisible",
    "A group of scientists accidentally opened a portal to another dimension",
    "The future of self-driving cars depends on the reliability of AI decision-making",
    "The traffic on my way to work this morning was unbearable",
    "The Eiffel Tower is one of the most famous landmarks in the world",
    "The clock struck midnight, and the entire town vanished",
    "He struggled to find the right words to express his gratitude",
    "Natural language processing allows chatbots to understand human emotions better",
    "The aroma of freshly brewed coffee filled the cozy café",
    "Sleep deprivation can negatively impact cognitive function and productivity",
    "The Great Wall of China was originally built to protect against invasions",
    "Web development and data science are two of the most popular tech fields today",
    "Artificial intelligence is expected to revolutionize many industries in the coming years",
    "The enchanted forest was said to grant wishes to those who entered with pure intentions",
    "The human body has an incredible ability to heal itself under the right conditions"
]

total_itokens = 0
total_gtokens = 0
total_time = 0
batch_size = 16
max_new_token = 20
num_sentences = len(sentences)

data_loader = DataLoader(sentences, batch_size=batch_size, shuffle=True)


# warmup run 
input_texts = sentences[10:27]
input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)

ttft_start = time.time()
with torch.no_grad():
    _ = quantized_model.generate(**input_ids, max_new_tokens=1, do_sample=True, temperature=0.7, repetition_penalty=1.5)
ttft_end = time.time()

# ttft run 
input_texts = sentences[5:22]
input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)

ttft_start = time.time()
with torch.no_grad():
    _ = quantized_model.generate(**input_ids, max_new_tokens=1, do_sample=True, temperature=0.7, repetition_penalty=1.5)
ttft_end = time.time()
    
for batch in data_loader:
    # tokenization and it's time
    tok_start = time.time()
    input_ids = tokenizer(batch, padding=True, return_tensors="pt")
    tok_time = time.time() - tok_start
    input_token = input_ids["input_ids"].numel()

    start_time = time.time()
    with torch.no_grad():
        output = quantized_model.generate(
            **input_ids,
            # input_ids["input_ids"],
            # attention_mask=input_ids["attention_mask"],
            max_new_tokens=max_new_token,
            # num_beams=4,
            do_sample = True,
            temperature = 0.7,
            repetition_penalty = 1.5
            )
    end_time = time.time()
    
    dec_start = time.time()
    res = res = [tokenizer.decode(op, skip_special_tokens=True) for op in output]
    dec_time = time.time() - dec_start

    num_tokens_generated = sum([o.shape[-1] for o in output])
    total_itokens += input_token
    total_gtokens += num_tokens_generated

    batch_total_time = tok_time + (end_time - start_time) + dec_time
    total_time += batch_total_time
    # for op in output: 
    #     print(tokenizer.decode(op, skip_special_tokens=True)+"\n")
    # print(output)
    # print(output.shape)
    # print(tokenizer.decode(output[1], skip_special_tokens=True))
total_tokens = total_gtokens - total_itokens
ttft = ttft_end - ttft_start
tpot = ((end_time - start_time) - ttft) / (total_tokens - 1)
latency = total_time / num_sentences
tps = total_tokens / total_time
rps = batch_size / latency

# Performance Measure
print("#"*10 + "  Performance Measure  " + "#"*10)
print(f" Number of generated tokens: {total_tokens}")
print(f" Batch Size: {batch_size}")
print(f" Latency: {latency} s")
print(f" TTFT: {ttft} s")
print(f" TPOT: {tpot} spt")
print(f" Throughput(TPS): {tps} tps")
print(f" Throughput(RPS): {rps} rps")
print(quantized_model)