import os
import tarfile

DATUMS_PATH = os.getenv("DATUMS_PATH", None)
DATASET_NAME = os.getenv("DATASET_NAME", None)

DATA_DIR = os.path.join(DATUMS_PATH, DATASET_NAME)
target_dir = '/tmp/dataset/'

def extract():
    print(DATA_DIR)
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('tar.gz')]
    print(files)
    for filename in files:
        print(filename)
        tar = tarfile.open(filename)
        tar.extractall(target_dir)
        tar.close()
    print("Extracted objects and stored. Location: ", target_dir)

if __name__ == '__main__':
    extract()
