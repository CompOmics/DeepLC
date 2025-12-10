
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import tf2onnx
from onnx2torch import convert
import torch

deeplc_dir = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODELS = [
    "mods/full_hc_PXD005573_pub_1fd8363d9af9dcad3be7553c39396960.keras",
    "mods/full_hc_PXD005573_pub_8c22d89667368f2f02ad996469ba157e.keras",
    "mods/full_hc_PXD005573_pub_cb975cfdd4105f97efa0b3afffe075cc.keras",
]
DEFAULT_MODELS = [os.path.join(deeplc_dir, dm) for dm in DEFAULT_MODELS]
def _convert_to_onnx():
    for model_path in DEFAULT_MODELS:
        if os.path.exists(model_path):
            mod = load_model(model_path)
            spec = [
                tf.TensorSpec([None, 60, 6], tf.float32, name="input_1"),
                tf.TensorSpec([None, 30, 6], tf.float32, name="input_2"),
                tf.TensorSpec([None, 55], tf.float32, name="input_3"),
                tf.TensorSpec([None, 60, 20], tf.float32, name="input_4"),
            ]
            onnx_model, _ = tf2onnx.convert.from_keras(mod, input_signature=spec, opset=13)
            torch_model = convert(onnx_model)
            torch.save(torch_model, model_path.replace(".keras", ".pt"))

if __name__ == "__main__":
    _convert_to_onnx()