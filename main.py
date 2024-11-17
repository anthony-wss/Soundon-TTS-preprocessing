from argparse import ArgumentParser
import os
from multiprocessing import Process, Queue, set_start_method
import queue
from module.worker import PreprocessWorker
import time

def worker_func(audio_root_dir, text_root_dir, worker_id, channel_id_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    worker = PreprocessWorker(audio_root_dir, text_root_dir)
    while True:
        try:
            channel_id = channel_id_queue.get_nowait()
            worker.run_channel(channel_id)
        except queue.Empty:
            break
        else:
            time.sleep(.5)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("--audio_root_dir", required=True)
    parser.add_argument("--text_root_dir", required=True)
    parser.add_argument("--num_worker", type=int, default=1)
    args = parser.parse_args()

    channel_id_queue = Queue()
    for channel_id in os.listdir(args.audio_root_dir):
        channel_id_queue.put(channel_id)

    number_of_processes = args.num_worker
    processes = []
    for w in range(number_of_processes):
        p = Process(target=worker_func, args=(args.audio_root_dir, args.text_root_dir, w, channel_id_queue))
        processes.append(p)
        p.start()

    # Completing process
    for p in processes:
        p.join()

