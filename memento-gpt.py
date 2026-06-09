import sentencepiece as spm
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

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

batch_size = 4 # number of independent sequences that will be processed in parallel
block_size = 8 # maximum context length for predictions

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch("train")
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # B (batch), T (time), C (channel)
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
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
    # No saved model yet -> train one, then save it so future runs reuse it.
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size = 32
    for steps in range(10000):
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

# The model is now ready (freshly trained or loaded) -- generate / experiment below.
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))
