import six.moves.urllib as urllib
import os
import tarfile
import shutil

OUTPUT_DIR = "/opt/dkube/output"

def download():
    opener = urllib.request.URLopener()
    opener.retrieve("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", OUTPUT_DIR + "/annotations.tar.gz")
    opener.retrieve("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", OUTPUT_DIR + "/images.tar.gz")
    print("Downloaded and saved the dataset. Location: ", OUTPUT_DIR)

if __name__ == '__main__':
    download()