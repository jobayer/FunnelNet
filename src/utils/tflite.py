import os
import tensorflow as tf


def tf_lite_model(features, model_dir):
    features_cast = tf.cast(features, tf.float32)
    features_ds = tf.data.Dataset.from_tensor_slices(features_cast).batch(1)

    def _representative_data_gen():
        for input_value in features_ds.take(100):
            yield [input_value]

    if os.path.isdir(model_dir):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    else:
        keras_model = tf.keras.models.load_model(model_dir)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()
