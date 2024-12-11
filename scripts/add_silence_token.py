from datasets import load_from_disk
import os
import csv
from module.wordframe import WordFrame
from math import ceil
from argparse import ArgumentParser


def add_silence_mask(example):
    trans_filename = example["user_audio_path"].split("/")[-1][:-3] + "csv"
    timestamp_file = os.path.join(TRANS_DIR, trans_filename)
    # print("DEBUG", timestamp_file)
    frame_list = []
    with open(timestamp_file) as fin:
        reader = csv.reader(fin)
        next(reader, None)

        for row in reader:
            frame = WordFrame(
                speaker = int(row[0]),
                start_sec = float(row[1][:-1]),
                end_sec = float(row[2][:-1]),
                content = row[3]
            )
            frame_list.append(frame)

    start = 0 
    while True:
        """
        Mask the machine's unit with -100 when it's user speaking

        Example:
        1, t1, t2, 你
        1, t3, t4, 好
        0, t5, t6, 機
        0, t7, t8, 器
        0, t9, t10, 人
        0, t11, t12, 你
        0, t13, t14, 好
        1, t15, t16, 請
        1, t17, t18, 問

        start_idx = ceil(t4 * 12.5) + 1
        end_idx   = ceil(t15 * 12.5) - 1
        """
        while start+1 < len(frame_list) and frame_list[start].speaker == 1:
            start += 1
        if start+1 >= len(frame_list):
            break

        end = start + 1
        while end+1 < len(frame_list) and frame_list[end].speaker == 0:
            end += 1

        start_idx = ceil(frame_list[start].end_sec * 12.5) + 1
        if start == 0:
            start_idx = 0
        end_idx   = min(len(example["machine_unit"][0]), ceil(frame_list[end].start_sec   * 12.5) - 1)

        # print("DEBUG", f"{start_idx} to {end_idx} is masked.")

        for i in range(8):
            for j in range(start_idx, end_idx+1):
                example["machine_unit"][i][j] = -100

        start = end + 1
    return example

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trans_dir", required=True, help="Path to timestamp/")
    parser.add_argument("--hf_dataset_path", required=True, help="Path to hf_dialogue_chinese_llama31_70B_user_long_2/")
    args = parser.parse_args()

    if args.hf_dataset_path[-1] == '/':
        args.hf_dataset_path = args.hf_dataset_path[:-1]

    ds = load_from_disk(args.hf_dataset_path)
    ds.map(add_silence_mask).save_to_disk(args.hf_dataset_path + "_with_silence")

