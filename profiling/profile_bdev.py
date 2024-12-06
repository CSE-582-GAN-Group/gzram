import subprocess
from pathlib import Path
import sys
import re

WRITE_TIME_PATTERN = re.compile(rb'write time: (.*)')
READ_TIME_PATTERN = re.compile(rb'read time: (.*)')

if len(sys.argv) != 2:
    print('usage: needs path')
    exit(1)

bdev_name = sys.argv[1]
input_path = Path('/home/cc/silesia_1G/dickens')
bdev_path = f'/dev/{bdev_name}'

sizes = []
times = []

for exponent in range(12, 30+2, 2):
    bs = 2**exponent
    proc = subprocess.run(f"sudo ../build/tests/test_write_read_file {bdev_path} {input_path} {bs}", shell=True, stdout=subprocess.PIPE)
    # time = WRITE_TIME_PATTERN.match(proc.stdout).group(1)
    # print(proc.stdout)
    time = READ_TIME_PATTERN.search(proc.stdout).group(1)
    times.append(float(time))

print(times)