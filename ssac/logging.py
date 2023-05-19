import json
import os
from collections import defaultdict
from queue import Queue
from threading import Thread
from typing import Any

import cloudpickle
import numpy as np
from numpy import typing as npt
from tabulate import tabulate
from tensorboardX import SummaryWriter

from ssac import metrics as m


class TrainingLogger:
    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir, flush_secs=60)
        self._metrics = defaultdict(m.MetricsAccumulator)
        self.log_dir = log_dir

    def __getitem__(self, item: str):
        return self._metrics[item]

    def __setitem__(self, key: str, value: float):
        self._metrics[key].update_state(value)

    def flush(self):
        self._writer.flush()

    def log_summary(self, summary: dict[str, float], step: int, flush: bool = False):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)
        with open(os.path.join(self.log_dir, "summary.jsonl"), "a") as file:
            file.write(json.dumps({"step": step, **summary}) + "\n")
        if flush:
            self._writer.flush()

    def log_metrics(self, step: int, flush: bool = False):
        table = []
        if len(self._metrics) == 0:
            return
        for k, v in self._metrics.items():
            metrics = v.result
            self._writer.add_scalar(
                k,
                float(metrics.mean),
                step,
            )
            result = v.result
            table.append([k, result.mean, result.std, result.min, result.max])
            v.reset_states()
        print(
            tabulate(
                table,
                headers=["Metric", "Mean", "Std", "Min", "Max"],
                tablefmt="orgtbl",
            )
        )
        if flush:
            self._writer.flush()

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
        flush: bool = False,
    ):
        # (N, T, C, H, W)
        self._writer.add_video(
            name, np.array(images, copy=False).transpose([0, 1, 4, 2, 3]), step, fps=fps
        )
        if flush:
            self._writer.flush()

    def __getstate__(self):
        self._writer.close()
        self._metrics.clear()
        state = self.__dict__.copy()
        del state["_writer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._writer = SummaryWriter(self.log_dir)

    def close(self):
        self._writer.close()


class StateWriter:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.queue: Queue[bytes] = Queue(maxsize=5)
        self._thread = Thread(name="state_writer", target=self._worker)
        self._thread.start()

    def write(self, data: dict[str, Any]):
        state_bytes = cloudpickle.dumps(data)
        self.queue.put(state_bytes)
        # Lazily open up a thread and let it drain the work queue. Thread exits
        # when there's no more work to do.
        if not self._thread.is_alive():
            self._thread = Thread(name="state_writer", target=self._worker)
            self._thread.start()

    def _worker(self):
        while not self.queue.empty():
            state_bytes = self.queue.get(timeout=1)
            with open(os.path.join(self.log_dir, "state.pkl"), "wb") as f:
                f.write(state_bytes)
                self.queue.task_done()

    def close(self):
        self.queue.join()
        if self._thread.is_alive():
            self._thread.join()
