import sentencepiece as spm
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # number of independent sequences that will be processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

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

# print(encode("hello world"))
# print(decode(encode("hello world")))

# encode the entire dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

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

# xb, yb = get_batch("train")
# print("inputs:")
# print(xb.shape)
# print(xb)
# print("targets:")
# print(yb.shape)
# print(yb)
# print("---")

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")

# print(xb)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
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
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
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
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        # B (batch), T (time), C (channel)
        
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
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
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)

# the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

MODEL_PATH = "model.pt"

if os.path.exists(MODEL_PATH):
    # A trained model already exists on disk -> load its weights and skip training.
    # (state_dict = just the learned tensors; we load them into a fresh model of the same shape.)
    m.load_state_dict(torch.load(MODEL_PATH))
    m.eval()
    print(f"loaded trained model from {MODEL_PATH} (skipped training)")
else:
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")
        
        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"final train loss: {loss.item():.4f}")
    torch.save(m.state_dict(), MODEL_PATH)   # save only the weights, not the whole object
    print(f"saved trained model to {MODEL_PATH}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
