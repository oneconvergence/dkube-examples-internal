# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source file include taxi pipeline functions and necesasry utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a
_build_estimator function needs to be provided.  This file contains both.
"""

import os  # pylint: disable=unused-import

import tensorflow as tf  # pylint: disable=unused-import

import tensorflow_transform as tft
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import schema_utils

import tensorflow_model_analysis as tfma

# 'id', 'Date_received', 'Sub_product', 'Issue', 'Sub_issue', 'Consumer_complaint_narrative',
# 'Company_public_response', 'Company', 'State', 'ZIP_code', 'Tags',
# 'Consumer_consent_provided', 'Submitted_via', 'Date_sent_to_company', 'Company_response_to_consumer',
# 'Timely_response', 'Consumer_disputed', 'Complaint_ID'

CHARACTERS = '%abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"\'?!.,:; *'
MAX_LEN = 50
embedding_dim = 10
num_characters = len(CHARACTERS)
vocab_size = num_characters + 3

# Categorical features are assumed to each have a maximum value in the dataset.
_MAX_CATEGORICAL_FEATURE_VALUES = [60, 4502, 18, 2]  # , 100000]

_CATEGORICAL_FEATURE_KEYS = [
        'State',
        'Company',
        'Product',
        'Timely_response',
        'Submitted_via',
]

_DENSE_FLOAT_FEATURE_KEYS = [
    # 'ZIP_code',
]

# # Number of buckets used by tf.transform for encoding each feature.
_FEATURE_BUCKET_COUNT = 10

_BUCKET_FEATURE_KEYS = [
]

# # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
_VOCAB_SIZE = 1000

# # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10

_VOCAB_FEATURE_KEYS = []

DELIMITERS = '.,!?() '

_TAG_FEATURE_KEYS = [
    # 'Tags',
]

_NLP_FEATURE_KEYS = [
    'Issue'
]

# Keys
_LABEL_KEY = 'Consumer_disputed'

# Step 4 START - -------------------------


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


# # Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn():
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def _fill_in_missing(x, to_string=False, unk=""):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = unk if x.dtype == tf.string else 0
    if to_string:
        default_value = tf.cast(default_value, tf.string)
    return tf.squeeze(
        tf.sparse_to_dense(x.indices, [x.dense_shape[0], 1], x.values,
                           default_value),
        axis=1)


labels = ['No', 'Yes']
table = tf.contrib.lookup.index_table_from_tensor(labels)


def _convert_label(x, table):
    # default_value = 'No'
    # x = tf.squeeze(tf.sparse_to_dense(
    #     x.indices, [x.dense_shape[0], 1], x.values, default_value),
    #     axis=1)  # TODO: Fix map '' to 'No'
    # vocab = tft.compute_and_apply_vocabulary(x, top_k=2)
    # return vocab
    # vocab = tf.sparse.to_dense(
    #     vocab.indices, [vocab.dense_shape[0], 1], vocab.values)
    # return tf.squeeze(vocab, axis=1)

    # one_hot = tf.one_hot(table.lookup(x), num_labels)
    # return tf.reshape(one_hot, [-1, num_labels])
    value = table.lookup(x)
    # return value
    # value = tf.sparse.to_dense(
    #     value.indices, [value.dense_shape[0], 1], value.values)
    return tf.reshape(value, [-1, 1])


# def convert_label_to_one_hot(x, table, num_labels):
#     # labels = ['No', 'Yes']
#     # num_labels = len(labels)
#     # table = tf.contrib.lookup.index_table_from_tensor(labels)

#     dense_x = tf.sparse_tensor_to_dense(x, default_value=0)
#     dense_vector = table.lookup(dense_x)
#     # dense_vector = tf.sparse.to_dense(
#     #     spare_vector.indices, [spare_vector.dense_shape[0], 1], spare_vector.values)
#     one_hot = tf.one_hot(dense_vector, num_labels)
#     return tf.reshape(one_hot, [-1, num_labels])


def _convert_tags(x, max_num_vocab=100):
    _tokens = tf.compat.v1.string_split(x, DELIMITERS)
    _indices = tft.compute_and_apply_vocabulary(_tokens, top_k=max_num_vocab)
    # Add one for the oov bucket created by compute_and_apply_vocabulary.
    # TODO: Using the _weights
    _bow_indices, _ = tft.tfidf(_indices, max_num_vocab + 1)
    return _bow_indices


def convert_string_to_indices(input_string):
    input_characters = tf.string_split(input_string, delimiter="")
    input_characters_dense = tf.sparse.to_dense(
        input_characters, default_value="*")
    mapping_characters = tf.string_split([CHARACTERS], delimiter="")
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_characters.values, default_value=num_characters+2, num_oov_buckets=1)
    sparse_tensor = tf.SparseTensor(
        input_characters.indices,
        table.lookup(input_characters.values),
        input_characters.dense_shape)
    return sparse_tensor, input_characters_dense


def pad_tensor(char_indices_sparse_tensor, input_characters_dense, max_len=MAX_LEN):
    paddings = [[0, 0], [0, tf.maximum(
        0, max_len-tf.shape(input_characters_dense)[1])]]
    tensor_padded = tf.pad(tf.sparse_tensor_to_dense(
        char_indices_sparse_tensor, default_value=num_characters+1),
        paddings, 'CONSTANT', constant_values=num_characters+1)
    return tensor_padded


def convert_character_to_indices(input_string, max_len=MAX_LEN):
    # slice at max length
    input_string = tf.strings.substr(input_string, 0, max_len)
    # split the string in characters and convert to indices
    char_indices_sparse_tensor, input_characters_dense = convert_string_to_indices(
        input_string)
    # pad the tensor if needed
    tensor_padded = pad_tensor(
        char_indices_sparse_tensor, input_characters_dense)
    # tensor_padded.set_shape([-1, max_len])

    print("SHAPE >>>>>", tensor_padded)
    return tf.reshape(tensor_padded, [-1, max_len])


def convert_label(label, table, num_labels):

    dense_x = tf.sparse_tensor_to_dense(label, default_value='No')
    dense_vector = table.lookup(dense_x)
    one_hot = tf.one_hot(dense_vector, num_labels)
    sparse_tensor = tf.SparseTensor(
        one_hot.indices,
        one_hot.values,
        one_hot.dense_shape)
    # return tf.reshape(one_hot, [-1, num_labels])
    return sparse_tensor


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, setting nan's to the mean.
        outputs[_transformed_name(key)] = tft.scale_to_0_1(
            _fill_in_missing(inputs[key]))

    for key in _VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

    for key in _TAG_FEATURE_KEYS:
        outputs[_transformed_name(key)] = _convert_tags(
            _fill_in_missing(inputs[key], unk="UNK"),
            max_num_vocab=20)

    for key in _NLP_FEATURE_KEYS:
        outputs[_transformed_name(key)] = convert_character_to_indices(
            _fill_in_missing(inputs[key], unk=""))

    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key], to_string=True),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

    # Complaint raised?
    # outputs[_transformed_name(_LABEL_KEY)] = _convert_label(inputs[_LABEL_KEY])
    labels = ['No', 'Yes']
    # num_labels = len(labels)
    table = tf.contrib.lookup.index_table_from_tensor(labels)

    # outputs[_transformed_name(_LABEL_KEY)] = _convert_label(
    #     inputs[_LABEL_KEY], table)
    # For the label column we provide the mapping from string to index.
    # labels = ['No', 'Yes']
    # table = tf.contrib.lookup.index_table_from_tensor(labels)

    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[_LABEL_KEY]), top_k=2, num_oov_buckets=0)

    # outputs[_transformed_name(_LABEL_KEY)] = \
    #     convert_label_to_one_hot(inputs[_LABEL_KEY], table, len(labels))

    # TODO: Lookup instead of vocab
    # outputs[_transformed_name(_LABEL_KEY)] = _convert_label(
    # _fill_in_missing(inputs[_LABEL_KEY]), table)

    # labels = inputs[inputs[_LABEL_KEY]]
    # _ = tft.uniques(labels, vocab_filename=_transformed_name(_LABEL_KEY))

    return outputs


# Step 4 END - -------------------------

# Step 5 START - -------------------------


def get_model(config, show_summary=True):
    input_state = tf.keras.layers.Input(shape=(1,), name="State_xf")
    input_company = tf.keras.layers.Input(shape=(1,), name="Company_xf")
    input_product = tf.keras.layers.Input(shape=(1,), name="Product_xf")
    input_timely_response = tf.keras.layers.Input(
        shape=(1,), name="Timely_response_xf")
    # input_zip_code = tf.keras.layers.Input(shape=(1,), name="ZIP_code_xf")
    # input_tags = tf.keras.layers.Input(shape=(1,), name="Tags_xf")
    # input_issue = tf.keras.layers.Input(shape=(1,), name="Issue_xf")
    input_submitted_via = tf.keras.layers.Input(
        shape=(1,), name="Submitted_via_xf")

    x0 = tf.keras.layers.Embedding(60, 5)(input_state)
    x0 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x0)

    x1 = tf.keras.layers.Embedding(2500, 20)(input_company)
    x1 = tf.keras.layers.Reshape((20, ), input_shape=(1, 20))(x1)

    x2 = tf.keras.layers.Embedding(2, 2)(input_product)
    x2 = tf.keras.layers.Reshape((2, ), input_shape=(1, 2))(x2)

    x3 = tf.keras.layers.Embedding(2, 2)(input_timely_response)
    x3 = tf.keras.layers.Reshape((2, ), input_shape=(1, 2))(x3)

    # x4 = tf.keras.layers.Embedding(10000, 25)(input_zip_code)

    # PROBLEM: [(None, 1, 5), (None, 1, 20), (None, 1, 2), (None, 1, 2), (None, 1), (None, 1, 3)]
    # Pad the input

    # WEIGHTS

    # x5 = keras.layers.Reshape(target_shape)(input_tags)
    # x6 = input_issue
    x7 = tf.keras.layers.Embedding(10, 3)(input_submitted_via)
    x7 = tf.keras.layers.Reshape((3, ), input_shape=(1, 3))(x7)

    conv_input = tf.keras.layers.Input(shape=(MAX_LEN, ), name="Issue_xf")
    conv_x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(conv_input)
    conv_x = tf.keras.layers.Conv1D(128, 5, activation='relu')(conv_x)
    conv_x = tf.keras.layers.GlobalMaxPooling1D()(conv_x)
    conv_x = tf.keras.layers.Dense(10, activation='relu')(conv_x)

    # reshape
    # conv_x = tf.keras.layers.Reshape((1, 10))(conv_x)
    # conv_x = tf.keras.layers.Reshape((1, 10), input_shape=(10,))(conv_x)
    # input_zip_code = tf.keras.layers.Reshape((1, 1))(input_zip_code)

    x_feed_forward = tf.keras.layers.concatenate(
        [x0, x1, x2, x3, x7, conv_x])
    x_feed_forward = conv_x
    x = tf.keras.layers.Dense(100, activation='relu')(x_feed_forward)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output = tf.keras.layers.Dense(
        1, activation='sigmoid', name='Consumer_disputed_xf')(x)
    inputs = [input_state, input_company, input_product, input_timely_response,
              input_submitted_via, conv_input]
    # inputs = [conv_input]
    tf_model = tf.keras.models.Model(inputs, output)
    tf_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',  # categorical_crossentropy
                     metrics=['accuracy'])
    if show_summary:
        tf_model.summary()

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=tf_model, config=config)
    return estimator


def _example_serving_receiver_fn(transform_output, schema):
    """Build the serving in inputs.

    Args:
      transform_output: directory in which the tf-transform model was written
        during the preprocessing step.
      schema: the schema of the input data.

    Returns:
      Tensorflow graph which parses examples, applying tf-transform to them.
    """
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            os.path.join(transform_output, transform_fn_io.TRANSFORM_FN_DIR),
            serving_input_receiver.features))

    _ = transformed_features.pop(_transformed_name(_LABEL_KEY))

    # for i in transformed_features:
    #     if i in ('Company_xf', 'Product_xf'):
    #         continue
    #     transformed_features.pop(i)

    print(">>>>>", transformed_features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(transform_output, schema):
    """Build everything needed for the tf-model-analysis to run the model.

    Args:
      transform_output: directory in which the tf-transform model was written
        during the preprocessing step.
      schema: the schema of the input data.

    Returns:
      EvalInputReceiver function, which contains:
        - Tensorflow graph which parses raw untransformed features, applies the
          tf-transform preprocessing operators.
        - Set of raw, untransformed features.
        - Label against which predictions will be compared.
    """
    # Notice that the inputs are raw features, not transformed features here.
    raw_feature_spec = _get_raw_feature_spec(schema)

    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')

    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.parse_example(serialized_tf_example, raw_feature_spec)

    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            os.path.join(transform_output, transform_fn_io.TRANSFORM_FN_DIR),
            features))

    # The key name MUST be 'examples'.
    receiver_tensors = {'examples': serialized_tf_example}

    # NOTE: Model is driven by transformed features (since training works on the
    # materialized output of TFT, but slicing will happen on raw features.
    features.update(transformed_features)

    f = {}
    for k in features.keys():
        if "_xf" in k and k != "Consumer_disputed_xf":
            f.update({k: features[k]})
        # features.pop(k)

    print(">>>> FEATURES", f)

    return tfma.export.EvalInputReceiver(
        features=f,  # features,
        receiver_tensors=receiver_tensors,
        labels=tf.reshape(transformed_features[_transformed_name(_LABEL_KEY)], [-1, 1]))


def _input_fn(filenames, transform_output, batch_size=2):
    """Generates features and labels for training or evaluation.

    Args:
      filenames: [str] list of CSV files to read data from.
      transform_output: directory in which the tf-transform model was written
        during the preprocessing step.
      batch_size: int First dimension size of the Tensors returned by input_fn

    Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
    """
    metadata_dir = os.path.join(transform_output,
                                transform_fn_io.TRANSFORMED_METADATA_DIR)
    transformed_metadata = metadata_io.read_metadata(metadata_dir)
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

    transformed_features = tf.contrib.learn.io.read_batch_features(
        filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

    # We pop the label because we do not want to use it as a feature while we're
    # training.

    transformed_labels = tf.reshape(transformed_features.pop(
        _transformed_name(_LABEL_KEY)), [-1, 1])
    return transformed_features, transformed_labels


# TFX will call this function
def trainer_fn(hparams, schema):
    """Build the estimator using the high level API.

    Args:
      hparams: Holds hyperparameters used to train the model as name/value pairs
      schema: Holds the schema of the training examples.

    Returns:
      A dict of the following:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.
    """
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    train_batch_size = 40
    eval_batch_size = 40

    def train_input_fn(): return _input_fn(  # pylint: disable=g-long-lambda
        hparams.train_files,
        hparams.transform_output,
        batch_size=train_batch_size)

    def eval_input_fn(): return _input_fn(  # pylint: disable=g-long-lambda
        hparams.eval_files,
        hparams.transform_output,
        batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
        train_input_fn,
        max_steps=hparams.train_steps)

    def serving_receiver_fn(): return _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
        hparams.transform_output, schema)

    exporter = tf.estimator.FinalExporter(
        'complaint_classification', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='complaint_classification-eval')

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=999, keep_checkpoint_max=1)

    run_config = run_config.replace(model_dir=hparams.serving_model_dir)

    # estimator = _build_estimator(
    #     transform_output=hparams.transform_output,

    #     # Construct layers sizes with exponetial decay
    #     hidden_units=[
    #         max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
    #         for i in range(num_dnn_layers)
    #     ],
    #     config=run_config,
    #     warm_start_from=hparams.warm_start_from)

    estimator = get_model(config=run_config)

    # Create an input receiver for TFMA processing
    def receiver_fn(): return _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
        hparams.transform_output, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }


# Step 5 END - -------------------------
