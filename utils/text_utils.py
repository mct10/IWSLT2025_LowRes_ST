import subprocess
from pathlib import Path


def count_lines(filename: Path) -> int:
    result = subprocess.run(["wc", "-l", filename], stdout=subprocess.PIPE)
    return int(result.stdout.decode().split()[0])
