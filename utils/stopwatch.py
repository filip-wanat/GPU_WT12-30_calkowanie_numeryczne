import time


class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def current_timer(self):
        return time.perf_counter() - self.start


if __name__ == "__main__":
    stopwatch = Stopwatch()
    time.sleep(1)
    print(stopwatch.current_timer())
