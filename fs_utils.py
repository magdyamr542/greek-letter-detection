from pathlib import Path
import os


def delete_dir_content(path: str, log=False):
    try:
        if len(os.listdir(path)) > 0:
            [f.unlink() for f in Path(path).iterdir() if f.is_file()]
            if log:
                print(f"delete content of dir {path}")
    except:
        print(f"dir {path} does not exist")
