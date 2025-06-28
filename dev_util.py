import os


def clear_folder(folder):
    ls = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    for filename in ls:
        os.remove(filename)
