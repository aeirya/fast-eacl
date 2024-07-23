def count_files(dir):
    import os

    if not os.path.exists(dir):
        return 0

    files = os.listdir(dir)
    # filter system files
    files = list(filter(lambda path: path[0] != '.', files))
    
    return len(files)
