from datasets import load_from_disk
from transformers import AutoTokenizer
import random
import torch
from module.mimimodel import MimiModel
import soundfile as sf


TOKENIZER_REPO = "voidful/Llama-3.2-11B-ASR"

ds = load_from_disk("moshi_tts_dataset/8199eec0-9a0f-4561-9053-ee1bddddccd1")

sample_ids = list(range(len(ds)))
random.shuffle(sample_ids)
x_vectors = []

mimi = MimiModel()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

for i in range(10):
    row = ds[sample_ids[i]]
    print("text:", row['text'])
    x_vectors.append(torch.tensor(row['x-vector']))
    units = torch.tensor(row['unit'])
    assert units.shape[1] == len(tokenizer.encode(row['text_with_pad'], add_special_tokens=False))
    audio_values = mimi.decode(units)
    sf.write(f"output-{i}.wav", audio_values.squeeze(), 24000)

cosine_sim = torch.nn.CosineSimilarity(dim=-1)
for i in range(10):
    for j in range(10):
        print(round(float(cosine_sim(x_vectors[i], x_vectors[j])), 2), end="\t")
    print()

