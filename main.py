import csv
import soundfile as sf
import torchaudio
from transformers.modeling_tf_utils import parse
import moshi
import torch
import soundfile as sf
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import random

from module.wordframe import WordFrame
from module.mimimodel import MimiModel
from module.xvectormodel import XVectorModel
from module.audiodatamanager import AudioDataManager
from module.textaligner import TextAligner

from argparse import ArgumentParser
import time
from copy import deepcopy


TOKENIZER_REPO = "voidful/Llama-3.2-11B-ASR"
PAD_TOKEN = "[PAD]"
PAD_TOKEN_ID = 51866
EPAD_TOKEN = "[END_PAD]"
PAD_TOKEN_ID = 51867
XVECTOR_SR = 16000
MIMI_SR = 24000
BATCH_SIZE = 16
    

mimi = MimiModel()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
xvector = XVectorModel()
audio_data_manager = AudioDataManager()
text_aligner = TextAligner(tokenizer, PAD_TOKEN, EPAD_TOKEN)


def dump_sample(batch_frame_list, audio_file, speaker_id, sample_path):
    global mimi
    global tokenizer
    global audio_data_manager
    global xvector
    global text_aligner

    # Step 1: Extract x-vector
    batch_audio = []
    for frame_list in batch_frame_list:
        start_sec = frame_list[0].start_sec
        end_sec = frame_list[-1].end_sec

        audio = audio_data_manager.get(audio_file, speaker_id, XVECTOR_SR)
        audio = audio[int(start_sec*XVECTOR_SR):int(end_sec*XVECTOR_SR)]
        batch_audio.append(audio)
    # sf.write(sample_path, audio, 16000) # data should be 1-dim tensor
    batch_spk_emb = xvector.encode(batch_audio, XVECTOR_SR)

    # Step 2: Extract mimi codec
    batch_audio = []
    for frame_list in batch_frame_list:
        start_sec = frame_list[0].start_sec
        end_sec = frame_list[-1].end_sec
        audio = audio_data_manager.get(audio_file, speaker_id, MIMI_SR)
        audio = audio[int(start_sec*MIMI_SR):int(end_sec*MIMI_SR)]
        batch_audio.append(audio)
    batch_codes = mimi.encode(batch_audio, MIMI_SR)

    # Step 3: Process Text
    batch_raw_text, batch_text_with_pad = [], []
    for i in range(len(batch_codes)):
        codes = batch_codes[i]
        frame_list = batch_frame_list[i]
        codec_length = codes.shape[-1]
        raw_text, text_with_pad = text_aligner.pad(frame_list, codec_length)
        batch_raw_text.append(raw_text)
        batch_text_with_pad.append(text_with_pad)
    
    sample_obj = {
        "text": batch_raw_text,
        "text_with_pad": batch_text_with_pad,
        "unit": batch_codes,
        "x-vector": batch_spk_emb
    }

    return sample_obj


def sort_duration(clips):
    return sorted(clips, key=lambda x: x.end_sec - x.start_sec)


def main():
    trans_file = "/work/u3937558/Soundon-TTS/dev/tw_transcription/word/8199eec0-9a0f-4561-9053-ee1bddddccd1/25fa8eda-af76-4637-8820-f10fe5166e31.csv"
    # audio_file = "/work/u3937558/Soundon-TTS/dev/tw_separated/8199eec0-9a0f-4561-9053-ee1bddddccd1/25fa8eda-af76-4637-8820-f10fe5166e31.m4a"
    audio_file = "../MMLM/source.wav"

    speaker_set = set()
    with open(trans_file) as csvfile:
        start_time = time.time()
        reader = csv.reader(csvfile)
        next(reader, None)

        # Step 1: read all word frames
        frame_list = []
        for row in reader:
            frame = WordFrame(
                speaker = int(row[0]),
                start_sec = float(row[1][:-1]),
                end_sec = float(row[2][:-1]),
                content = row[3]
            )
            speaker_set.add(int(row[0]))
            frame_list.append(frame)


        # Step 2: concatenate frames into samples. Split at silence >= 0.3 second.
        clips = {spk: [] for spk in speaker_set}
        for frame in frame_list:
            spk_clips = clips[frame.speaker]
            # Only run at the first frame of every speaker
            if not spk_clips:
                spk_clips.append([frame])
            if frame.start_sec - spk_clips[-1][-1].end_sec >= 0.3:
                spk_clips.append([frame])
            else:
                spk_clips[-1].append(frame)

        # for spk in speaker_set:
        #     for clip in clips[spk]:
        #         print(clip[0].start_sec, clip[-1].end_sec, clip[-1].end_sec - clip[0].start_sec)
        end_time = time.time()
        print("[debug] Step 1, 2", end_time - start_time)


        # Step 3: extract unit and x-vector
        dataset_dict = {"unit": [], "x-vector": [], "text": [], "text_with_pad": []}
        for spk in speaker_set:
            # patient = 10
            idx = 0
            spk_dataset_dict = {"unit": [], "x-vector": [], "text": [], "text_with_pad": []}
            batch_clip = []
            clips[spk].sort(key=lambda clip: clip[-1].end_sec - clip[0].start_sec)
            for clip in tqdm(clips[spk]):
                if clip[-1].end_sec - clip[0].start_sec < 0.3:
                    continue
                batch_clip.append(clip)

                if len(batch_clip) == BATCH_SIZE:
                    sample_obj = dump_sample(batch_clip, audio_file, speaker_id=spk, sample_path=f"./output-{idx}.wav")
                # idx += 1
                    spk_dataset_dict["text"].extend(sample_obj["text"]) 
                    spk_dataset_dict["text_with_pad"].extend(sample_obj["text_with_pad"]) 
                    spk_dataset_dict["unit"].extend(sample_obj["unit"]) 
                    spk_dataset_dict["x-vector"].extend(sample_obj["x-vector"]) 
                    batch_clip = []

                # patient -= 1
                # if patient < 0:
                #     break
            random.shuffle(spk_dataset_dict["x-vector"])
            for colname in ["unit", "x-vector", "text", "text_with_pad"]:
                dataset_dict[colname].extend(spk_dataset_dict[colname])

        ds = Dataset.from_dict(dataset_dict) 
        ds.save_to_disk("moshi_tts_dataset_dummy")
        # ds.push_to_hub("anthony-wss/moshi_tts_dataset_dummy")    
        print(ds)
            # print(', '.join(row))

if __name__ == "__main__":
    # parser = ArgumentParser()
    # args = parser.parse_args()
    main()

