import queue
import threading

class InputManager:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.thread = threading.Thread(target=self._get_input, args=())
        self.thread.daemon = True
        self.thread.start()

    def _get_input(self):
        while True:
            inp = input()
            self.input_queue.put(inp)

    def get_input(self, blocking=False):
        if blocking:
            return self.input_queue.get()  # This will block until input is available
        else:
            try:
                return self.input_queue.get_nowait()  # This won't block
            except queue.Empty:
                return None
