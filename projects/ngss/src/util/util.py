from pathlib import Path



def make_and_return_dir(dir):
    if isinstance(dir, str):
        dir = Path(dir)
    else:
        pass

    dir.mkdir(parents=True, exist_ok=True)
    return dir




        




