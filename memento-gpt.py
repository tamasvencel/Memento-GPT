import sentencepiece as spm
import os
import torch

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
print(data.shape, data.dtype)
print(data[:1000])