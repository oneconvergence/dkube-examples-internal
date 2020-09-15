import os
import json
import multiprocessing
import tensorflow as tf
import tensorflow_hub as hub
import zipfile
import tarfile
import argparse
from tensorflow.python.ops import metrics as metrics_lib
from dkube import dkubeLoggerHook as logger_hook
from tensorflow.python.platform import tf_logging as logging

tf.compat.v1.disable_eager_execution() 

tf.compat.v1.logging.info('TF Version {}'.format(tf.__version__))
tf.compat.v1.logging.info('GPU Available {}'.format(tf.test.is_gpu_available()))
if 'TF_CONFIG' in os.environ:
    tf.compat.v1.logging.info('TF_CONFIG: {}'.format(os.environ["TF_CONFIG"]))

FLAGS = None
DATA_DIR = "/opt/dkube/input"
MODEL_DIR = "/opt/dkube/output"
TFHUB_CACHE_DIR = os.getenv('TFHUB_CACHE_DIR', "/opt/dkube/input")
BATCH_SIZE = int(os.getenv('BATCHSIZE', 10))
EPOCHS = int(os.getenv('EPOCHS', 1))
TF_TRAIN_STEPS = int(os.getenv('STEPS',1000))
summary_interval = 100
print ("TF_CONFIG: {}".format(os.getenv("TF_CONFIG", '{}')))

steps_epoch  = 0
#if not os.path.isdir(MODEL_DIR):
#    os.makedirs(MODEL_DIR)

def count_epochs(iterator):
    if os.getenv('TF_CONFIG', None) == None:
        return
    cluster_spec = json.loads(os.getenv('TF_CONFIG',None))
    role = cluster_spec['task']
    host = cluster_spec['cluster'][role['type']][role['index']]
    if len(cluster_spec['cluster'].keys()) > 1:
     sess = tf.compat.v1.Session('grpc://'+ host)
    else:
     sess = tf.compat.v1.Session()
    global steps_epoch
    if not steps_epoch:
        while True:
            try:
                sess.run(iterator)
                steps_epoch += 1
            except Exception as OutOfRangeError:
                if steps_epoch == 0:
                   steps_epoch = TF_TRAIN_STEPS
                steps_epoch /= FLAGS.num_epochs
                break

def _img_string_to_tensor(image_string, image_size=(299, 299)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize(image_decoded_as_float, size=image_size)
    
    return image_resized

def make_input_fn(file_pattern, image_size=(299, 299), shuffle=False, batch_size=BATCH_SIZE, num_epochs=EPOCHS, buffer_size=4096):
    
    def _path_to_img(path):
        # Get the parent folder of this file to get it's class name
        label = tf.compat.v1.string_split([path], delimiter='/').values[-2]
        
        # Read in the image from disk
        image_string = tf.io.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size)
        
        return { 'inputs': image_resized }, label
    
    def _input_fn():
        dataset = tf.data.Dataset.list_files(file_pattern)
        if shuffle:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size, num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=_path_to_img, batch_size=FLAGS.batch_size))
        (images, labels) = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        (cimages, clabels) = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        count_epochs(cimages)
        return (images, labels)

    return _input_fn

def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    NUM_CLASSES = len(params['label_vocab'])

    module = hub.Module(TFHUB_CACHE_DIR, trainable=is_training and params['train_module'], name=params['module_name'])
    bottleneck_tensor = module(features['inputs'])

    with tf.compat.v1.name_scope('final_retrain_ops'):
        logits = tf.compat.v1.layers.dense(bottleneck_tensor, units=1, trainable=is_training)

    def train_op_fn(loss):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        return optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    if NUM_CLASSES == 2:
        head = tf.estimator.BinaryClassHead(label_vocabulary=params['label_vocab'])
    else:
        head = tf.estimator.MultiClassHead(n_classes=NUM_CLASSES, label_vocabulary=params['label_vocab'])

    spec =  head.create_estimator_spec(
        features, mode, logits, labels, train_op_fn=train_op_fn
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.compat.v1.summary.scalar('accuracy', metrics_lib.accuracy(labels, spec.predictions['classes'])[1])
        logging_hook = logger_hook({"loss": spec.loss,"accuracy":
            tf.compat.v1.metrics.accuracy(labels, spec.predictions['classes'])[1], 
            "step" : tf.compat.v1.train.get_or_create_global_step(), "steps_epoch": steps_epoch, "mode":"train"}, every_n_iter=summary_interval)
        spec = spec._replace(training_hooks = [logging_hook])
    if mode == tf.estimator.ModeKeys.EVAL:
        print("evaluating")
        logging_hook = logger_hook({"loss": spec.loss, "accuracy":
            spec.eval_metric_ops['accuracy'][0], "step" : 
            tf.compat.v1.train.get_or_create_global_step(), "steps_epoch": steps_epoch, "mode": "eval"}, every_n_iter=summary_interval)
        spec = spec._replace(evaluation_hooks = [logging_hook])
    return spec

def train(_):
    try:
      fp = open(os.getenv('DKUBE_JOB_HP_TUNING_INFO_FILE', 'None'),'r')
      hyperparams = json.loads(fp.read())
      hyperparams['num_epochs'] = EPOCHS
    except:
      hyperparams = { "learning_rate":1e-3, "batch_size":BATCH_SIZE, "num_epochs":EPOCHS }
      pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=float(hyperparams['learning_rate']), help='Learning rate for training.')
    parser.add_argument('--batch_size', type=int, default=int(hyperparams['batch_size']), help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=int(hyperparams['num_epochs']), help='Number of epochs to train for.')
    global FLAGS, DATA_DIR
    FLAGS, unparsed = parser.parse_known_args()
    run_config = tf.estimator.RunConfig(model_dir=MODEL_DIR, save_summary_steps=summary_interval, save_checkpoints_steps=summary_interval)
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
        EXTRACT_PATH = "/tmp/tfhub-cache-dir"
        files = [os.path.join(TFHUB_CACHE_DIR, f) for f in tf.io.gfile.listdir(TFHUB_CACHE_DIR) if f.endswith('tar.gz')]
        for fname in files:
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(EXTRACT_PATH)
            tar.close()
            TFHUB_CACHE_DIR = EXTRACT_PATH
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
    train_input_fn = make_input_fn(train_files, image_size=input_img_size, shuffle=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=TF_TRAIN_STEPS)

    eval_files = os.path.join(DATA_DIR, 'valid', '**/*.jpg')
    eval_input_fn = make_input_fn(eval_files, image_size=input_img_size, shuffle=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=1, throttle_secs=1, start_delay_secs=1)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    def serving_input_receiver_fn():
        feature_spec = {
                'inputs': tf.io.FixedLenFeature([], dtype=tf.string)
        }
        default_batch_size = 1
        serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string,shape=[default_batch_size],name='input_image_tensor')
        received_tensors = {'inputs':serialized_tf_example}
        features = tf.io.parse_example(serialized=serialized_tf_example, features=feature_spec)
        fn = lambda image: _img_string_to_tensor(image, input_img_size)
        features['inputs'] = tf.map_fn(fn, features['inputs'], dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver(features, received_tensors)
    if os.getenv('TF_CONFIG', '') != '':
        config = json.loads(os.getenv('TF_CONFIG'))
        if config['task']['type'] == 'master':
            classifier.export_saved_model(MODEL_DIR, serving_input_receiver_fn)
    else:
        classifier.export_saved_model(MODEL_DIR, serving_input_receiver_fn)


def run():
    global summary_interval
    summary_interval = 100
    if TF_TRAIN_STEPS%100 < 10 and TF_TRAIN_STEPS < 1000:
        summary_interval = TF_TRAIN_STEPS/10
    if TF_TRAIN_STEPS <= 100:
        summary_interval = 10;
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=train)

if __name__ == '__main__':
    if os.getenv("STEPS") is None:
        os.environ['STEPS'] = str(TF_TRAIN_STEPS)
    run()
