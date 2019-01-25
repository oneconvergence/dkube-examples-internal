import os
import multiprocessing
import tensorflow as tf
import tensorflow_hub as hub
import zipfile
import tarfile
from tensorflow.python.ops import metrics as metrics_lib
from dkube import dkubeLoggerHook as logger_hook
from tensorflow.python.platform import tf_logging as logging

tf.logging.info('TF Version {}'.format(tf.__version__))
tf.logging.info('GPU Available {}'.format(tf.test.is_gpu_available()))
if 'TF_CONFIG' in os.environ:
    tf.logging.info('TF_CONFIG: {}'.format(os.environ["TF_CONFIG"]))

DATUMS_PATH = os.getenv('DATUMS_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', None)
MODEL_DIR = os.getenv('OUT_DIR', None)
TFHUB_CACHE_DIR = os.getenv('MODEL_PATH',None)
BATCH_SIZE = int(os.getenv('TF_BATCH_SIZE', 64))
EPOCHS = int(os.getenv('TF_EPOCHS', 1))
TF_TRAIN_STEPS = int(os.getenv('TF_TRAIN_STEPS',1000))
summary_interval = 100
if TFHUB_CACHE_DIR != None:
    if not os.path.isdir(os.path.join(TFHUB_CACHE_DIR, 'resnet_v2_50')):
	TFHUB_CACHE_DIR = None
    else:
	TFHUB_CACHE_DIR = os.path.join(TFHUB_CACHE_DIR, 'resnet_v2_50')

print ("TF_CONFIG: {}".format(os.getenv("TF_CONFIG", '{}')))

steps_epoch  = 0
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def count_epochs(iterator):
    sess = tf.Session()
    global steps_epoch
    if not steps_epoch:
        while True:
            try:
                sess.run(iterator)
                steps_epoch += 1
            except Exception as OutOfRangeError:
                steps_epoch /= EPOCHS
                break

def _img_string_to_tensor(image_string, image_size=(299, 299)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    
    return image_resized

def make_input_fn(file_pattern, image_size=(299, 299), shuffle=False, batch_size=BATCH_SIZE, num_epochs=EPOCHS, buffer_size=4096):
    
    def _path_to_img(path):
        # Get the parent folder of this file to get it's class name
        label = tf.string_split([path], delimiter='/').values[-2]
        
        # Read in the image from disk
        image_string = tf.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size)
        
        return { 'inputs': image_resized }, label
    
    def _input_fn():
        dataset = tf.data.Dataset.list_files(file_pattern)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_path_to_img, batch_size=BATCH_SIZE))
        (images, labels) = dataset.make_one_shot_iterator().get_next()
        (cimages, clabels) = dataset.make_one_shot_iterator().get_next()
        count_epochs(cimages)
        return (images, labels)

    return _input_fn

def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    NUM_CLASSES = len(params['label_vocab'])

    module = hub.Module(TFHUB_CACHE_DIR, trainable=is_training and params['train_module'], name=params['module_name'])
    bottleneck_tensor = module(features['inputs'])

    with tf.name_scope('final_retrain_ops'):
        logits = tf.layers.dense(bottleneck_tensor, units=1, trainable=is_training)

    def train_op_fn(loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        return optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if NUM_CLASSES == 2:
        head = tf.contrib.estimator.binary_classification_head(label_vocabulary=params['label_vocab'])
    else:
        head = tf.contrib.estimator.multi_class_head(n_classes=NUM_CLASSES, label_vocabulary=params['label_vocab'])

    spec =  head.create_estimator_spec(
        features, mode, logits, labels, train_op_fn=train_op_fn
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        logging_hook = logger_hook({"loss": spec.loss,"accuracy":
            metrics_lib.accuracy(labels, spec.predictions['classes'])[1], 
            "step" : tf.train.get_or_create_global_step(), "steps_epoch": steps_epoch, "mode":"train"}, every_n_iter=summary_interval)
        spec = spec._replace(training_hooks = [logging_hook])
    if mode == tf.estimator.ModeKeys.EVAL:
        logging_hook = logger_hook({"loss": spec.loss, "accuracy":
            spec.eval_metric_ops['accuracy'][1], "step" : 
            tf.train.get_or_create_global_step(), "steps_epoch": steps_epoch, "mode": "eval"}, every_n_iter=summary_interval)
        spec = spec._replace(evaluation_hooks = [logging_hook])
    return spec

def train(_):
    
    run_config = tf.estimator.RunConfig(model_dir=MODEL_DIR, save_summary_steps=summary_interval, save_checkpoints_steps=summary_interval)
    DATA_DIR = "{}/{}".format(DATUMS_PATH, DATASET_NAME)
    print ("ENV, EXPORT_DIR:{}, DATA_DIR:{}".format(MODEL_DIR, DATA_DIR))
    EXTRACT_PATH = "/tmp/resnet-model"
    ZIP_FILE = DATA_DIR + "/data.zip"
    if os.path.exists(ZIP_FILE):
        print("Extracting compressed training data...")
        archive = zipfile.ZipFile(ZIP_FILE)
        for file in archive.namelist():
            if file.startswith('data'):
                archive.extract(file, EXTRACT_PATH)
        print("Training data successfuly extracted")
        DATA_DIR = EXTRACT_PATH + "/data"    

    params = {
        'module_spec': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
        'module_name': 'resnet_v2_50',
        'learning_rate': 1e-3,
        'train_module': False,  # Whether we want to finetune the module
        'label_vocab': os.listdir(os.path.join(DATA_DIR, 'valid'))
    }
    global TFHUB_CACHE_DIR
    if TFHUB_CACHE_DIR != None:
        files = [os.path.join(TFHUB_CACHE_DIR, f) for f in tf.gfile.ListDirectory(TFHUB_CACHE_DIR) if f.endswith('tar.gz')]
        for fname in files:
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(TFHUB_CACHE_DIR)
            tar.close()
    else:
        TFHUB_CACHE_DIR = params['module_spec']

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        config=run_config,
        params=params
    )

    input_img_size = hub.get_expected_image_size(hub.Module(TFHUB_CACHE_DIR))

    train_files = os.path.join(DATA_DIR, 'train', '**/*.jpg')
    train_input_fn = make_input_fn(train_files, image_size=input_img_size, shuffle=True)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=TF_TRAIN_STEPS)

    eval_files = os.path.join(DATA_DIR, 'valid', '**/*.jpg')
    eval_input_fn = make_input_fn(eval_files, image_size=input_img_size)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=1, throttle_secs=1, start_delay_secs=1)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    def serving_input_receiver_fn():
        feature_spec = {
                'inputs': tf.FixedLenFeature([], dtype=tf.string)
        }
        default_batch_size = 1
        serialized_tf_example = tf.placeholder(dtype=tf.string,shape=[default_batch_size],name='input_image_tensor')
        received_tensors = {'inputs':serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        fn = lambda image: _img_string_to_tensor(image, input_img_size)
        features['inputs'] = tf.map_fn(fn, features['inputs'], dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver(features, received_tensors)

    classifier.export_savedmodel(MODEL_DIR, serving_input_receiver_fn)

def run():
    global summary_interval
    summary_interval = 100
    if TF_TRAIN_STEPS%100 < 10 and TF_TRAIN_STEPS < 1000:
        summary_interval = TF_TRAIN_STEPS/10
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=train)

if __name__ == '__main__':
    run()
