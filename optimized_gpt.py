import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import pickle
import time

# Optimized hyperparameters
batch_size = 256  # Increased from 64 for better GPU utilization
block_size = 256
max_iters = 3000  # Reduced since we're training more efficiently
eval_interval = 300  # Less frequent evaluation
learning_rate = 1e-3  # Slightly increased
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100  # Reduced for faster evaluation
n_embd = 256  # Reduced from 384 for faster training on small dataset
n_head = 4    # Reduced from 6
n_layer = 4   # Reduced from 6
dropout = 0.2

# GPU optimizations
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device}")
print(f"Batch size: {batch_size}")
print(f"Model size: {n_embd} embedding dim, {n_head} heads, {n_layer} layers")

torch.manual_seed(1337)

# Load and preprocess data
print("Loading data...")
try:
    with open('bg.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Error: 'bg.txt' file not found. Please make sure the file exists in the current directory.")
    print("You can download a sample dataset like:")
    print("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    exit(1)

# Character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare data and move to GPU immediately
print("Preparing data...")
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

def get_batch(split):
    """Optimized batch generation directly on GPU"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    """Evaluate model performance"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    """Single self-attention head"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU generally works better than ReLU for transformers
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block with pre-norm architecture"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """Optimized GPT Language Model"""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text with optional temperature sampling"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            with autocast():
                logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print("Initializing model...")
    model = GPTLanguageModel()
    
    # Compile model for PyTorch 2.0+ (comment out if using older PyTorch)
    if hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    model = model.to(device)
    
    # Print model info
    param_count = count_parameters(model) / 1e6
    print(f"Model parameters: {param_count:.2f}M")
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if device == 'cuda' else None
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    print("Starting training...")
    start_time = time.time()
    
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {elapsed:.1f}s")
        
        # Training step with mixed precision
        xb, yb = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:  # CUDA with mixed precision
            with autocast():
                logits, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # CPU or without mixed precision
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/3600:.2f}h)")
    
    # Generate sample text
    print("\nGenerating sample text...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    sample_text = decode(generated[0].tolist())
    print("\nGenerated text:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    # Save model and vocabulary
    print("Saving model...")
    model_state = model.state_dict()
    if hasattr(model, '_orig_mod'):  # Handle compiled models
        model_state = model._orig_mod.state_dict()
    
    torch.save(model_state, 'gpt_optimized.pth')
    
    with open('vocab.pkl', 'wb') as f:
        pickle.dump((stoi, itos), f)
    
    print("Model saved to 'gpt_optimized.pth'")
    print("Vocabulary saved to 'vocab.pkl'")