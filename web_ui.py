import torch
import gradio as gr
from gpt import GPTLanguageModel  # this should import your model class

# Load vocab if needed
import pickle
with open('vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)


encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt.pth', map_location=device))
model.to(device)
model.eval()

# Inference function
def generate_text(prompt, max_tokens=200):
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens)
    return decode(output[0].tolist())

# Launch Gradio UI
gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Slider(50, 1000, step=50, value=200, label="Max Tokens")
    ],
    outputs="text",
    title="BG GPT Generator",
    description="Enter a prompt and generate text using your trained GPT model.",
).launch()
