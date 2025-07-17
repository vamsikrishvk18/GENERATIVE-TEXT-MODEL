!pip install transformers
!pip install torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model_name = "gpt2"  # You can also use 'gpt2-medium', 'gpt2-large', etc.

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()
# Input prompt
prompt = "Once upon a time in a small village"

# Tokenize input and convert to tensor
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(
    inputs,
    max_length=100,       # Total output length (including prompt)
    num_return_sequences=1,  # Number of different outputs
    no_repeat_ngram_size=2,  # Avoid repeating phrases
    do_sample=True,       # Enable sampling (for creativity)
    top_k=50,             # Consider top 50 tokens
    top_p=0.95,           # Nucleus sampling
    temperature=0.9       # Higher = more random
)

# Decode and print result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
