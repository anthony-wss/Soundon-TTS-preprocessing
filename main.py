from argparse import ArgumentParser
import os
from multiprocessing import Process, Queue, set_start_method
import queue
from module.worker import PreprocessWorker
from module.datasets.libritts import LibriTTSLoader
import time
import torch
from datasets import Dataset

def worker_func(split_name, worker_id, audio_queue):
    torch.cuda.set_device(worker_id)
    worker = PreprocessWorker()
    dataset_dict = {"unit": [], "x-vector": [], "text": [], "text_with_pad": []}
    while True:
        try:
            sample = audio_queue.get_nowait()
            sample_dict = worker.run_sample(sample)
            if sample_dict:
                for key in ["unit", "x-vector", "text", "text_with_pad"]:
                    dataset_dict[key].extend(sample_dict[key])
        except queue.Empty:
            break
        else:
            time.sleep(.5)
    ds = Dataset.from_dict(dataset_dict) 
    ds.save_to_disk(os.path.join("libritts_dataset", split_name, str(worker_id)))

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    set_start_method("spawn", force=True)

    parser = ArgumentParser()

    # TODO: Make Soundon works by defining `datasets/soundon.py`
    parser.add_argument("--audio_root_dir")
    parser.add_argument("--text_root_dir")
    parser.add_argument("--num_worker", type=int, default=1)
    args = parser.parse_args()

    split_name="dev-clean"
    dataset = LibriTTSLoader(root_dir="/home/anthony/disk-1t/libritts_pure_audio", split_name=split_name)
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

