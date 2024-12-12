import sys
from pathlib import Path

SIZE = 2**30

def repeat(file_path: Path, output_file_path: Path, target_size: int):
    with file_path.open('rb') as f:
        contents = f.read()

    with output_file_path.open('wb') as f_out:
        while f_out.tell() < target_size:
            f_out.write(contents)
            if f_out.tell() > target_size:
                f_out.truncate(target_size)

def main():
    if len(sys.argv) != 3:
        print("usage: python expand_files.py <source_folder> <output_folder>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.glob('*'):
        output_file_path = output_path / file_path.name

        file_size = file_path.stat().st_size
        print(f"Expanding {file_path} to {output_file_path}")
        if file_size < SIZE:
            repeat(file_path, output_file_path, SIZE)
        else:
            output_file_path.write_bytes(file_path.read_bytes())

if __name__ == "__main__":
    main()
