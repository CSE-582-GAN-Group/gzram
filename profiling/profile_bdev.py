import subprocess
from pathlib import Path

bdev_name = 'zram0'
input_file = '/home/cc/dickens_1G'

bdev_sys = Path('/sys/block') / bdev_name

class TimeTracker:    
    def __init__(self):
        self.last_time = self.read_time()

    def read_time(self):
        path = bdev_sys / 'stat'
        stats = path.read_text().split()
        return int(stats[9])

    def get_time_elapsed(self):
        time = self.read_time()
        elapsed = time - self.last_time
        self.last_time = time
        return elapsed

time_tracker = TimeTracker()
times = []

for exponent in range(12, 30):
    bs = 2 ** exponent
    subprocess.run(f"sudo dd if={input_file} of=/dev/{bdev_name} bs={bs} count=1 oflag=direct", shell=True)
    time = time_tracker.get_time_elapsed()
    times.append(time)

print(times)