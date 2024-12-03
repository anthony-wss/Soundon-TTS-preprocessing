# import huggingface_hub
# huggingface_hub.login()
from datasets import load_from_disk, concatenate_datasets, DatasetDict
import os
from tqdm import tqdm

libritts = {}
for split_name in tqdm(os.listdir("./libritts_dataset")):
    whole_dataset = None
    try:
        for i in range(8):
            ds = load_from_disk(os.path.join("./libritts_dataset/", split_name, str(i)))

            if not whole_dataset:
                whole_dataset = ds
            else:
                whole_dataset = concatenate_datasets([whole_dataset, ds])
    except Exception as e:
        print(str(e))
        continue
    
    libritts[split_name] = whole_dataset
    # whole_dataset.save_to_disk(f"./libritts_dataset/{split_name}")

dataset_dict = DatasetDict(libritts)
dataset_dict.save_to_disk(f"./hf_LibriTTS")
# whole_dataset.push_to_hub("anthony-wss/Soundon-tts")

