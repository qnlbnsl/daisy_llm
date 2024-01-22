import queue
import threading
from typing_extensions import Self


class InputManager:
    def __init__(self: Self):
        self.input_queue: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._get_input, args=())
        self.thread.daemon = True
        self.thread.start()

    def _get_input(self: Self) -> None:
        while True:
            inp = input()
            self.input_queue.put(inp)

    def get_input(self, blocking: bool = False) -> str | None:
        if blocking:
            return self.input_queue.get()  # This will block until input is available
        else:
            try:
                return self.input_queue.get_nowait()  # This won't block
            except queue.Empty:
                return None
