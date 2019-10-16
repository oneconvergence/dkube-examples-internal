import os
import tarfile

DATA_DIR = os.getenv('DKUBE_INPUT_DATASETS', None)
if DATA_DIR is not None:
    DATA_DIR = DATA_DIR.split(",")[0]
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
