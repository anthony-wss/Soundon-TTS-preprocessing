# import huggingface_hub
# huggingface_hub.login()
from datasets import load_from_disk, concatenate_datasets, DatasetDict
import os
from tqdm import tqdm

# libritts = {}
# for split_name in tqdm(os.listdir("./dialogue_chinese_llama31_70B_user_long_2")):
#     whole_dataset = None
#     try:
whole_dataset = None
for i in range(8):
    print(i)
    ds = load_from_disk(os.path.join("./dialogue_chinese_llama31_70B_user_long_2/", str(i)))

    if not whole_dataset:
        whole_dataset = ds
    else:
        whole_dataset = concatenate_datasets([whole_dataset, ds])
whole_dataset.save_to_disk(f"./hf_dialogue_chinese_llama31_70B_user_long_2")
exit()
    # except Exception as e:
    #     print(str(e))
    #     continue
    # 
    # libritts[split_name] = whole_dataset
    # whole_dataset.save_to_disk(f"./libritts_dataset/{split_name}")

dataset_dict = DatasetDict(libritts)
dataset_dict.save_to_disk(f"./hf_dialogue_chinese_llama31_70B_user_long_2")
# whole_dataset.push_to_hub("anthony-wss/Soundon-tts")

