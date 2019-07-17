# Convert the keras model format to tensorflow model format for serving
import tensorflow as tf
import os

out_dir = os.getenv('OUT_DIR', None)
save_dir = "/tmp"
model_name = 'keras_mnist_trained_model.h5'
 
# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
model = tf.keras.models.load_model(os.path.join(save_dir, model_name))
export_path = os.path.join(save_dir, '1')

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})
        
# Move the saved model to s3
tf.gfile.Rename(export_path, out_dir)
