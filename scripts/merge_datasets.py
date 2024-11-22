import huggingface_hub
huggingface_hub.login(token="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")
from datasets import load_from_disk, concatenate_datasets
import os
from tqdm import tqdm

whole_dataset = None
for ds_dir in tqdm(os.listdir("./moshi_tts_dataset")):
    try:
        ds = load_from_disk(os.path.join("./moshi_tts_dataset/", ds_dir))

        if not whole_dataset:
            whole_dataset = ds
        else:
            whole_dataset = concatenate_datasets([whole_dataset, ds])
    except Exception as e:
        print(str(e))
        continue
    
print(whole_dataset)
whole_dataset.push_to_hub("anthony-wss/Soundon-tts")

