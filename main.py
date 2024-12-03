from argparse import ArgumentParser
import os
from multiprocessing import Process, Queue, set_start_method
import queue
from module.worker import PreprocessWorker
from module.datasets.libritts import LibriTTSLoader
import time
import torch
from datasets import Dataset
import logging


def worker_func(split_name, worker_id, audio_queue):
    torch.cuda.set_device(worker_id)
    worker = PreprocessWorker()
    dataset_dict = {"unit": [], "x-vector": [], "text": [], "text_with_pad": []}

    logger = logging.getLogger(f"worker_{worker_id}")
    handler = logging.FileHandler(f'example.{worker_id}.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    while True:
        try:
            time.sleep(.5)
            sample = audio_queue.get_nowait()
            sample_dict = worker.run_sample(sample)
            if sample_dict:
                logger.debug(f"Processing sample: {sample[1]}")
                length = sum([len(x) for x in sample_dict["text"]])
                logger.debug(f"Content length: {length}")
                for key in ["unit", "x-vector", "text", "text_with_pad"]:
                    dataset_dict[key].extend(sample_dict[key])
        except queue.Empty:
            logger.info("Queue is empty, worker is exiting.")
            break
    ds = Dataset.from_dict(dataset_dict) 
    ds.save_to_disk(os.path.join("libritts_dataset", split_name, str(worker_id)))

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    set_start_method("spawn", force=True)

    parser = ArgumentParser()

    # TODO: Make Soundon works by defining `datasets/soundon.py`
    parser.add_argument("--audio_root_dir")
    parser.add_argument("--text_root_dir")
    # parser.add_argument("--split_name", required=True)
    parser.add_argument("--num_worker", type=int, default=1)
    args = parser.parse_args()


    split_name="dev-clean"
    for split_name in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    # for split_name in ["train-clean-100", "train-clean-360"]:
        dataset = LibriTTSLoader(root_dir="/mnt/home/ntuspeechlabtaipei1/anthony/Soundon-TTS-preprocessing/libritts_pure_audio", split_name=split_name)
        audio_queue = Queue()
        for sample in dataset.iterate():
            audio_queue.put(sample)

        number_of_processes = args.num_worker
        processes = []
        for w in range(number_of_processes):
            p = Process(target=worker_func, args=(split_name, w, audio_queue))
            processes.append(p)
            p.start()

        # Completing process
        for p in processes:
            p.join()

