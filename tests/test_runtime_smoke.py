"""Exercise genuine TensorFlow conversion and inference without shipping a model."""

import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from demo import Model
from predict import main as predict_images

try:
    import tensorflow as tf
except ImportError:
    tf = None

CONFIG = {
    "input_size": [3, 4, 4],
    "crop_pct": 1.0,
    "crop_mode": "center",
    "interpolation": "bicubic",
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
}


@unittest.skipIf(tf is None, "Install requirements.txt to test the real TFLite runtime")
class RuntimeSmokeTests(unittest.TestCase):
    @staticmethod
    def create_export(folder, mode):
        """Convert a deliberately tiny graph and describe its real output tensor."""

        class TinyModel(tf.Module):
            @tf.function(
                input_signature=[tf.TensorSpec([1, 4, 4, 3], tf.float32, name="images")]
            )
            def predict(self, images):
                if mode == "classification":
                    colors = tf.reduce_mean(images[..., :2], axis=[1, 2])
                    return {"probabilities": tf.nn.softmax(colors, axis=1)}
                background = tf.fill([1, 4, 4, 1], tf.constant(0.1, tf.float32))
                logits = tf.concat([background, images[..., :2]], axis=-1)
                return {"logits": tf.transpose(logits, [0, 3, 1, 2])}

        tiny = TinyModel()
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tiny.predict.get_concrete_function()], tiny
        )
        data = converter.convert()
        (folder / "model.tflite").write_bytes(data)
        # This test's metadata reader is not the beginner-facing Model class.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^\s*Warning: tf\.lite\.Interpreter is deprecated",
                category=UserWarning,
            )
            interpreter = tf.lite.Interpreter(model_content=data)
        interpreter.allocate_tensors()
        output_name = "probabilities" if mode == "classification" else "logits"
        output_index = interpreter.get_output_details()[0]["index"]
        manifest = {
            "task": {"mode": mode},
            "labels": ["red", "blue"],
            "preprocessing": {"timm_config": CONFIG},
            "inputs": {"images": {"layout": "NHWC"}},
            "tflite_sha256": hashlib.sha256(data).hexdigest(),
            "tflite": {"outputs": {output_name: {"index": int(output_index)}}},
        }
        if mode == "segmentation":
            manifest["segmentation"] = {"background_logit_bias": 0.0}
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_classification_runs_on_image_and_saves_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            export = folder / "export"
            export.mkdir()
            self.create_export(export, "classification")
            image = folder / "red.png"
            Image.new("RGB", (8, 4), "red").save(image)

            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                model = Model(export)
            self.assertFalse(
                any(
                    "tf.lite.Interpreter is deprecated" in str(item.message)
                    for item in observed
                )
            )
            crop, prediction = model.predict(Image.open(image))
            self.assertEqual(crop.size, (4, 4))
            self.assertEqual(prediction["scores"].shape, (2,))
            self.assertGreater(float(prediction["scores"][0]), 0.7)
            self.assertAlmostEqual(float(sum(prediction["scores"])), 1.0, places=5)

            output = folder / "results"
            predict_images([str(export), str(image), "--output", str(output)])
            self.assertTrue((output / "red.png.png").is_file())

    def test_segmentation_runs_on_image_and_saves_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            export = folder / "export"
            export.mkdir()
            self.create_export(export, "segmentation")
            image = folder / "red.png"
            Image.new("RGB", (8, 4), "red").save(image)

            model = Model(export)
            crop, prediction = model.predict(Image.open(image))
            self.assertEqual(crop.size, (4, 4))
            np.testing.assert_array_equal(
                prediction["mask"], np.zeros((4, 4), dtype=np.int64)
            )

            output = folder / "results"
            predict_images([str(export), str(image), "--output", str(output)])
            self.assertTrue((output / "red.png.png").is_file())
            np.testing.assert_array_equal(
                np.load(output / "red.png.npy"), prediction["mask"]
            )


if __name__ == "__main__":
    unittest.main()
