import six.moves.urllib as urllib
import os
<<<<<<< HEAD
import tensorflow as tf
=======
>>>>>>> parent of a7f2a26... - Copy to s3 using tf.gfile

OUTPUT_DIR = os.getenv('OUT_DIR', None)
opener = urllib.request.URLopener()
opener.retrieve("http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", OUTPUT_DIR + "/annotations.tar.gz")
print("Saved in", OUTPUT_DIR)
