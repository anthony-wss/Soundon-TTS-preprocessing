from module.mimimodel import MimiModel
from module.xvectormodel import XVectorModel
from module.audiodatamanager import AudioDataManager
from module.textaligner import TextAligner
from transformers import AutoTokenizer

from pydub import AudioSegment
import os.path as osp
import os
import csv
import numpy as np

import torch
from datasets import Dataset
from tqdm import tqdm
import random

from module.wordframe import WordFrame


TOKENIZER_REPO = "voidful/Llama-3.2-11B-ASR"
PAD_TOKEN = "[PAD]"
PAD_TOKEN_ID = 51866
EPAD_TOKEN = "[END_PAD]"
PAD_TOKEN_ID = 51867
XVECTOR_SR = 16000
MIMI_SR = 24000
# BATCH_SIZE_SEC = 300 # V100
BATCH_SIZE_SEC = 2000  # H100


class PreprocessWorker:

    # TODO: `audio_root_dir` and `text_root_dir` should be deprecated
    def __init__(self, audio_root_dir=None, text_root_dir=None):
        self.mimi = MimiModel()
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        self.xvector = XVectorModel()
        self.audio_data_manager = AudioDataManager()
        self.text_aligner = TextAligner(self.tokenizer, PAD_TOKEN, EPAD_TOKEN)

        self.audio_root_dir = audio_root_dir
        self.text_root_dir = text_root_dir


    def get_sample_duration(self):
        dur = float(np.random.normal(15, 5, 1)[0])
        dur = min(dur, 30)
        dur = max(dur, 1)
        return dur


    def dump_sample(self, batch_frame_list, audio_file, speaker_id):
        # Step 1: Extract x-vector
        batch_audio = []
        for frame_list in batch_frame_list:
            start_sec = frame_list[0].start_sec
            end_sec = frame_list[-1].end_sec

            audio = self.audio_data_manager.get(audio_file, speaker_id, XVECTOR_SR)
            audio = audio[int(start_sec*XVECTOR_SR):int(end_sec*XVECTOR_SR)]
            batch_audio.append(audio)
        # print([a.shape for a in batch_audio])
        # sf.write(sample_path, audio, 16000) # data should be 1-dim tensor
        batch_spk_emb = self.xvector.encode(batch_audio)

        # Step 2: Extract mimi codec
        batch_audio = []
        for frame_list in batch_frame_list:
            start_sec = frame_list[0].start_sec
            end_sec = frame_list[-1].end_sec
            audio = self.audio_data_manager.get(audio_file, speaker_id, MIMI_SR)
            audio = audio[int(start_sec*MIMI_SR):int(end_sec*MIMI_SR)]
            batch_audio.append(audio.numpy())
        batch_codes = self.mimi.encode(batch_audio)

        # Step 3: Process Text
        batch_raw_text, batch_text_with_pad = [], []
        idx_to_remove = []
        for i in range(len(batch_codes)):
            codes = batch_codes[i]
            frame_list = batch_frame_list[i]
            codec_length = codes.shape[-1]
            raw_text, text_with_pad = self.text_aligner.pad(frame_list, codec_length)
            if raw_text is None or text_with_pad is None:
                idx_to_remove.append(i)
                print("TextAligner fails at", audio_file)
            batch_raw_text.append(raw_text)
            batch_text_with_pad.append(text_with_pad)

        def remove_idx(arr, idx_to_remove):
            return [arr[i] for i in range(len(arr)) if i not in idx_to_remove]

        batch_raw_text = remove_idx(batch_raw_text, idx_to_remove)
        batch_text_with_pad = remove_idx(batch_text_with_pad, idx_to_remove)
        batch_codes = remove_idx(batch_codes, idx_to_remove)
        batch_spk_emb = remove_idx(batch_spk_emb, idx_to_remove)
        
        sample_obj = {
            "text": batch_raw_text,
            "text_with_pad": batch_text_with_pad,
            "unit": batch_codes,
            "x-vector": batch_spk_emb
        }

        return sample_obj


    def convert_to_wav(self, audio_file):
        """ Convert `audio_file` from m4a to wav, and return the path to the wav file. """
        # Ensure the file has a valid extension
        if audio_file.endswith(".wav"):
            return audio_file

        if not audio_file.lower().endswith(".m4a"):
            raise ValueError("Input file must be in .m4a format")
        
        wav_file = osp.splitext(audio_file)[0] + ".wav"
        
        audio = AudioSegment.from_file(audio_file, format="m4a")
        audio.export(wav_file, format="wav")
        
        return wav_file


    def run_conv_batch(self, samples):
        batch_xvector_audio = []
        batch_mimi_audio = []
        batch_frame_list = []
        for i in range(len(samples)):
            audio_file, trans_file = samples[i]

            try:
                audio_file = self.convert_to_wav(audio_file)
            except Exception as e:
                print("file error", audio_file, ":", str(e))
                continue

            frame_list = [[], []]
            with open(trans_file) as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)

                # Step 1: read all word frames
                for row in reader:
                    frame = WordFrame(
                        speaker = int(row[0]),
                        start_sec = float(row[1][:-1]),
                        end_sec = float(row[2][:-1]),
                        content = row[3]
                    )
                    frame_list[frame.speaker].append(frame)
            batch_frame_list.extend(frame_list)
            batch_xvector_audio.extend([
                self.audio_data_manager.get(audio_file, 0, XVECTOR_SR),
                self.audio_data_manager.get(audio_file, 1, XVECTOR_SR),
            ])
            batch_mimi_audio.append(
                self.audio_data_manager.get(audio_file, 1, MIMI_SR),
            )

        samples_dict = {"machine_unit": [], "x-vector": [], "text": [], "text_with_pad": [], "user_audio_path": []}
        batch_spk_emb = self.xvector.encode(batch_xvector_audio)
        batch_codes = self.mimi.encode(batch_mimi_audio)

        for i in range(0, len(batch_frame_list), 2):
            codes = batch_codes[i//2]
            samples_dict["machine_unit"].append(codes)
            codec_length = codes.shape[-1]

            # user
            u_raw_text, u_text_with_pad = self.text_aligner.pad(batch_frame_list[i], codec_length)
            # machine
            m_raw_text, m_text_with_pad = self.text_aligner.pad(batch_frame_list[i+1], codec_length)

            samples_dict["x-vector"].append([batch_spk_emb[i], batch_spk_emb[i+1]])
            samples_dict["text"].append([u_raw_text, m_raw_text])
            samples_dict["text_with_pad"].append([u_text_with_pad, m_text_with_pad])
            samples_dict["user_audio_path"].append(samples[i//2][0])

        return samples_dict


    def run_sample(self, sample):
        audio_file = sample[0]
        trans_file = sample[1]
        samples_dict = {"unit": [], "x-vector": [], "text": [], "text_with_pad": []}

        try:
            audio_file = self.convert_to_wav(audio_file)
        except:
            print("file not found", audio_file)
            return None

        speaker_set = set()
        with open(trans_file) as csvfile:
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
            spk_target_len = {spk: self.get_sample_duration() for spk in speaker_set}
            for frame in frame_list:
                spk_clips = clips[frame.speaker]
                # Only run at the first frame of every speaker
                if not spk_clips:
                    spk_clips.append([frame])
                    continue
                if frame.end_sec - spk_clips[-1][0].start_sec >= spk_target_len[frame.speaker]:
                    spk_clips.append([frame])
                    spk_target_len[frame.speaker] = self.get_sample_duration()
                else:
                    spk_clips[-1].append(frame)

            # for spk in speaker_set:
            #     for clip in clips[spk]:
            #         print(clip[0].start_sec, clip[-1].end_sec, clip[-1].end_sec - clip[0].start_sec)
            # end_time = time.time()
            # print("[debug] Step 1, 2", end_time - start_time)


            # Step 3: extract unit and x-vector
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

                    max_audio_sec = clip[-1].end_sec - clip[0].start_sec
                    if len(batch_clip) * max_audio_sec >= BATCH_SIZE_SEC:
                        sample_obj = self.dump_sample(batch_clip, audio_file, speaker_id=spk)
                    # idx += 1
                        spk_dataset_dict["text"].extend(sample_obj["text"]) 
                        spk_dataset_dict["text_with_pad"].extend(sample_obj["text_with_pad"]) 
                        spk_dataset_dict["unit"].extend(sample_obj["unit"]) 
                        spk_dataset_dict["x-vector"].extend(sample_obj["x-vector"]) 
                        batch_clip = []
                        torch.cuda.empty_cache()
                if batch_clip:
                    sample_obj = self.dump_sample(batch_clip, audio_file, speaker_id=spk)
                    spk_dataset_dict["text"].extend(sample_obj["text"]) 
                    spk_dataset_dict["text_with_pad"].extend(sample_obj["text_with_pad"]) 
                    spk_dataset_dict["unit"].extend(sample_obj["unit"]) 
                    spk_dataset_dict["x-vector"].extend(sample_obj["x-vector"])

                    # patient -= 1
                    # if patient < 0:
                    #     break
                random.shuffle(spk_dataset_dict["x-vector"])
                for colname in ["unit", "x-vector", "text", "text_with_pad"]:
                    samples_dict[colname].extend(spk_dataset_dict[colname])

        # os.remove(audio_file)
        self.audio_data_manager.pop(audio_file)
        torch.cuda.empty_cache()
        return samples_dict

    def run_channel(self, channel_id):
        dataset_dict = {"unit": [], "x-vector": [], "text": [], "text_with_pad": []}
        for audio_file in os.listdir(osp.join(self.audio_root_dir, channel_id)):
            if not audio_file.endswith(".m4a"):
                continue
            audio_id = audio_file[:-4]
            trans_file = osp.join(self.text_root_dir, channel_id, audio_id + ".csv")
            audio_file = osp.join(self.audio_root_dir, channel_id, audio_id + ".m4a")

            try:
                audio_file = self.convert_to_wav(audio_file)
            except:
                print("file not found", audio_file)
                continue

            speaker_set = set()
            with open(trans_file) as csvfile:
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
                spk_target_len = {spk: self.get_sample_duration() for spk in speaker_set}
                for frame in frame_list:
                    spk_clips = clips[frame.speaker]
                    # Only run at the first frame of every speaker
                    if not spk_clips:
                        spk_clips.append([frame])
                        continue
                    if frame.end_sec - spk_clips[-1][0].start_sec >= spk_target_len[frame.speaker]:
                        spk_clips.append([frame])
                        spk_target_len[frame.speaker] = self.get_sample_duration()
                    else:
                        spk_clips[-1].append(frame)

                # for spk in speaker_set:
                #     for clip in clips[spk]:
                #         print(clip[0].start_sec, clip[-1].end_sec, clip[-1].end_sec - clip[0].start_sec)
                # end_time = time.time()
                # print("[debug] Step 1, 2", end_time - start_time)


                # Step 3: extract unit and x-vector
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

                        max_audio_sec = clip[-1].end_sec - clip[0].start_sec
                        if len(batch_clip) * max_audio_sec >= BATCH_SIZE_SEC:
                            sample_obj = self.dump_sample(batch_clip, audio_file, speaker_id=spk)
                        # idx += 1
                            spk_dataset_dict["text"].extend(sample_obj["text"]) 
                            spk_dataset_dict["text_with_pad"].extend(sample_obj["text_with_pad"]) 
                            spk_dataset_dict["unit"].extend(sample_obj["unit"]) 
                            spk_dataset_dict["x-vector"].extend(sample_obj["x-vector"]) 
                            batch_clip = []
                            torch.cuda.empty_cache()


                        # patient -= 1
                        # if patient < 0:
                        #     break
                    random.shuffle(spk_dataset_dict["x-vector"])
                    for colname in ["unit", "x-vector", "text", "text_with_pad"]:
                        dataset_dict[colname].extend(spk_dataset_dict[colname])

            os.remove(audio_file)
            self.audio_data_manager.pop(audio_file)
            torch.cuda.empty_cache()
        ds = Dataset.from_dict(dataset_dict) 
        ds.save_to_disk(osp.join("moshi_tts_dataset", channel_id))
        # ds.push_to_hub("anthony-wss/moshi_tts_dataset_dummy")    
        print(ds)
        # print(', '.join(row))

