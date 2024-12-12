"""Program to profile a block device.
"""

import subprocess
from pathlib import Path
import sys
import re

WRITE_TIME_PATTERN = re.compile(rb'write time: (.*)')
READ_TIME_PATTERN = re.compile(rb'read time: (.*)')

if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} <device_name> <input_file>')
    exit(1)

bdev_name = sys.argv[1]
input_path = Path(sys.argv[2])
bdev_path = f'/dev/{bdev_name}'

NUM_TRIALS = 3

sizes = []
write_times = [[] for _ in range(NUM_TRIALS)]
read_times = [[] for _ in range(NUM_TRIALS)]

for exponent in range(12, 30+2, 2):
    bs = 2**exponent
    sizes.append(bs)
    for i in range(NUM_TRIALS):
        subprocess.run(f"sudo blkdiscard {bdev_path}", shell=True)
        proc = subprocess.run(f"sudo ../build/tests/test_write_read_file {bdev_path} {input_path} {bs}", shell=True, stdout=subprocess.PIPE)
        write_time = WRITE_TIME_PATTERN.match(proc.stdout).group(1)
        read_time = READ_TIME_PATTERN.search(proc.stdout).group(1)
        write_times[i].append(float(write_time))
        read_times[i].append(float(read_time))

print(f'Sizes')
print(sizes)
print(f'Write times (s)')
print(write_times)
print(f'Read times (s)')
print(read_times)
