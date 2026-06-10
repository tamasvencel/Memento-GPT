import sentencepiece as spm
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # number of independent sequences that will be processed in parallel
block_size = 128 # maximum context length for predictions
max_iters = 5000
eval_interval = 100 # evaluate often so we catch the best-val checkpoint before it overfits
learning_rate = 3e-4
warmup_iters = 100          # linear LR warmup steps before cosine decay (stabilizes early attention)
lr_decay_iters = max_iters  # cosine decays across the whole training run
min_lr = 3e-5               # LR floor at the end of decay (~learning_rate / 10)
patience = 5               # evals without val improvement before early stopping
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") # mixed precision only helps on the GPU
eval_iters = 50 # batches per loss estimate; 200 made evaluation dominate runtime (400 eval batches per 100 train steps)
# Model is deliberately small: with a fixed ~300k-token corpus, a large model overfits
# almost immediately (train loss -> 0.1 while val loss climbs). Less capacity + more dropout.
n_embd = 192
n_head = 6
n_layer = 4
dropout = 0.15

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high") # use TF32 matmuls on the GPU (faster, negligible accuracy cost)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Train the SentencePiece BPE tokenizer
if not os.path.exists("tok.model"):
    spm.SentencePieceTrainer.train(
        input="input.txt",          # corpus to learn the vocabulary from
        model_prefix="tok",         # output files: tok.model (the tokenizer) + tok.vocab (human-readable)
        vocab_size=4096,            # total tokens to learn. Sized to the corpus.
        model_type="bpe",           # Byte-Pair Encoding
        byte_fallback=True,         # unknown chars decompose into raw UTF-8 bytes instead of lossy <unk>
        character_coverage=1.0,     # cover 100% of characters (safe to do because byte_fallback handles the long tail)
        add_dummy_prefix=True,      # prepend a space to every input so "word" and " word" tokenize consistently
    )

# Load the trained tokenizer
sp = spm.SentencePieceProcessor(model_file="tok.model")

encode = sp.Encode          # str  -> list[int]
decode = sp.Decode          # list[int] -> str
vocab_size = sp.vocab_size()

# encode the entire dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train set, rest val
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp): # same precision as training
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- The original, didactic attention (kept for reference) ---
# One explicit head at a time: separate K/Q/V projections per head, a materialized
# (T, T) score matrix, a hand-applied causal mask from a `tril` buffer, then a Python
# loop in MultiHeadAttention concatenating the heads. Readable, but slow: 24 small
# matmuls per block where one big one would do, and every head stored its own
# (block_size, block_size) tril mask in the checkpoint.
#
# class Head(nn.Module):
#     """ one head of self-attention """
#
#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embd, head_size, bias=False)
#         self.query = nn.Linear(n_embd, head_size, bias=False)
#         self.value = nn.Linear(n_embd, head_size, bias=False)
#         self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         B,T,C = x.shape
#         k = self.key(x) # (B,T,head_size)
#         q = self.query(x) # (B,T,head_size)
#         # compute attention scores ("affinities")
#         wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
#         wei = F.softmax(wei, dim=-1) # (B, T, T)
#         wei = self.dropout(wei)
#         # perform the weighted aggregation of the values
#         v = self.value(x) # (B,T,head_size)
#         out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
#         return out
#
# class MultiHeadAttention(nn.Module):
#     """ multiple heads of self-attention in parallel """
#
#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out

class CausalSelfAttention(nn.Module):
    """ all heads in one fused pass (the production pattern, as in nanoGPT / GPT-2) """

    def __init__(self):
        super().__init__()
        # one matmul computes Q, K, V for every head at once
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2) # each (B, T, C)
        # split channels into heads: (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        # fused kernel (FlashAttention on GPU): causal mask, softmax, and attention
        # dropout happen inside - no materialized (T, T) matrix, no tril buffer
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble head outputs
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        # head_size = n_embd // n_head
        # self.sa = MultiHeadAttention(n_head, head_size) # old: explicit per-head loop
        self.sa = CausalSelfAttention()
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection around self-attention
        x = x + self.ffwd(self.ln2(x)) # residual connection around feed-forward
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # token + position embeddings feed a stack of Transformer blocks, then a final
        # layer norm and a linear head project to next-token logits over the vocabulary
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.lm_head = nn.Linear(n_embd, vocab_size) # old: separate output matrix
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying (GPT-2 / nanoGPT): input embedding and output head share one matrix.
        # Saves ~786k params (~23% of the model) - capacity this small corpus can't fill anyway.
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights) # GPT-2 style init (std=0.02) - REQUIRED once weights are tied

    def _init_weights(self, module):
        # Without this, the tied output head inherits nn.Embedding's default std=1, which makes the
        # init logits huge (init loss ~126 instead of ~ln(vocab)=8.3) and wastes ~1000 steps recovering.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # B (batch), T (time), C (channel)
        
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad() # sampling needs no gradients -> saves memory and time
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions (loss is None without targets)
            logits, _ = self(idx_cond)
            # focus only on the last time step, scaled by temperature (lower = more confident/repetitive)
            logits = logits[:, -1, :] / temperature # becomes (B, vocab_size)
            # optionally restrict sampling to the top_k most likely tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

MODEL_PATH = "model.pt"

if os.path.exists(MODEL_PATH):
    # A trained model already exists on disk -> load its weights and skip training.
    # (state_dict = just the learned tensors; we load them into a fresh model of the same shape.)
    # map_location lets a GPU-trained checkpoint load on a CPU-only machine;
    # weights_only is the safe way to load files you didn't just create yourself
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"loaded trained model from {MODEL_PATH} (skipped training)")
else:
    # weight decay only on the matmul weights (dim >= 2); biases and layernorm
    # params are excluded, as in nanoGPT / GPT-2
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=learning_rate)

    # learning-rate schedule: linear warmup then cosine decay (the transformer-idiomatic schedule)
    def get_lr(it):
        # 1) linear warmup for the first warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / warmup_iters
        # 2) then cosine-decay smoothly from learning_rate down to min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # goes 1 -> 0
        return min_lr + coeff * (learning_rate - min_lr)

    best_val = float("inf")   # best val loss seen -> checkpoint only when it improves
    patience_left = patience  # evals remaining before we stop early

    for step in range(max_iters):

        # set this step's learning rate from the warmup+cosine schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            if losses["val"] < best_val:
                # validation improved -> save best weights and refill patience
                best_val = losses["val"]
                patience_left = patience
                torch.save(model.state_dict(), MODEL_PATH) # save only the weights, not the whole object
                print(f"  new best val {best_val:.4f} -> saved {MODEL_PATH}")
            else:
                # no improvement -> spend one patience credit, stop when exhausted
                patience_left -= 1
                if patience_left == 0:
                    print(f"early stopping at step {step} (no val improvement in {patience} evals)")
                    break

        # sample a batch of data
        xb, yb = get_batch("train")

        # forward pass (bfloat16 autocast on GPU for speed + lower memory)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = model(xb, yb)

        # backward + update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients for training stability
        optimizer.step()

    print(f"training done. best val loss {best_val:.4f} (saved to {MODEL_PATH})")
    # current weights in memory are the LAST step -> reload the best checkpoint before generating
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

# generate from the model
model.eval() # disable dropout for inference
# seed from a real prompt: <unk> (id 0) renders as "?", and whitespace-only seeds normalize
# to empty, so we use a natural screenplay opener that the model has seen at the start of scenes.
context = torch.tensor([encode("INT.")], dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=200)[0].tolist()))
