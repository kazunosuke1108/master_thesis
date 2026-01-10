from glob import glob
import os
import sys
import traceback
from multiprocessing import cpu_count, Process, Queue
from datetime import datetime

# --- logging helper ---
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --- import Visualizer path ---
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
sys.path.append(os.path.expanduser("~") + "/kazu_ws/master_thesis/master_thesis_modules")
from scripts.visualize.visualizer_v5 import Visualizer

EXTRACT_DIR = "/media/hayashide/MobileSensing/Nagasaki20240827/jpg/extract"
PATTERN = EXTRACT_DIR + "/N*"

# --- glob with timeout to avoid infinite hang on CIFS ---
def _glob_worker(q: Queue, pattern: str):
    try:
        q.put(sorted(glob(pattern)))
    except Exception as e:
        q.put(e)

def safe_glob(pattern: str, timeout_sec: int = 60):
    q = Queue()
    p = Process(target=_glob_worker, args=(q, pattern))
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"glob() timed out after {timeout_sec}s: {pattern} (likely CIFS stall)")
    if q.empty():
        return []
    res = q.get()
    if isinstance(res, Exception):
        raise res
    return res

def create_mp4(event_dir_path: str, trial_dir_path: str):
    # 子プロセス側で Visualizer を作る（共有しない）
    try:
        visualizer = Visualizer(trial_name="20260110_findEvent", strage="NASK")
        event_name = os.path.basename(event_dir_path.rstrip("/"))
        jpg_paths = sorted(glob(os.path.join(event_dir_path, "*.jpg")))
        if not jpg_paths:
            log(f"[SKIP] no jpgs: {event_dir_path}")
            return
        mp4_path = os.path.join(trial_dir_path, f"{event_name}.mp4")
        visualizer.jpg2mp4(image_paths=jpg_paths, mp4_path=mp4_path, fps=20)
        log(f"[OK] {event_name} -> {mp4_path}")
    except Exception:
        log(f"[ERROR] failed: {event_dir_path}\n{traceback.format_exc()}")

def main():
    log("A")
    log(f"B EXTRACT_DIR={EXTRACT_DIR}")
    log(f"pattern={PATTERN}")
    log("C: before glob (safe_glob)")

    event_dir_paths = safe_glob(PATTERN, timeout_sec=120)
    log(f"D: after glob count={len(event_dir_paths)}")

    # 親側で trial_dir を決める（ここは軽い）
    visualizer_parent = Visualizer(trial_name="20260110_findEvent", strage="NASK")
    trial_dir_path = visualizer_parent.data_dir_dict["trial_dir_path"]
    os.makedirs(trial_dir_path, exist_ok=True)

    # プロセス数を抑える（CPU全開は事故りやすい）
    max_workers = min(8, cpu_count())  # 例: 上限6。好みで4〜8推奨
    log(f"max_workers={max_workers}")

    # バッチ起動
    p_list = []
    for i, event_dir_path in enumerate(event_dir_paths, 1):
        p = Process(target=create_mp4, args=(event_dir_path, trial_dir_path))
        p_list.append(p)
        log(f"Prepared {i}/{len(event_dir_paths)}: {event_dir_path}")

        if len(p_list) == max_workers or i == len(event_dir_paths):
            for p in p_list:
                p.start()
            for p in p_list:
                p.join()
            p_list = []

    log("DONE")

if __name__ == "__main__":
    main()
