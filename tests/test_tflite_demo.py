"""Fast synthetic TFLite tests: TensorFlow and model downloads are not required."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ijk_tflite_demo", ROOT / "demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)
with patch.dict(sys.modules, {"demo": demo}):
    PREDICT_SPEC = importlib.util.spec_from_file_location("ijk_tflite_predict", ROOT / "predict.py")
    predict = importlib.util.module_from_spec(PREDICT_SPEC)
    PREDICT_SPEC.loader.exec_module(predict)


CONFIG = {
    "input_size": [3, 4, 4],
    "crop_pct": 1.0,
    "crop_mode": "center",
    "interpolation": "bicubic",
    "mean": [0, 0, 0],
    "std": [1, 1, 1],
}


def make_export(folder, mode, bias=0.0):
    model = folder / "model.tflite"
    model.write_bytes(b"synthetic test artifact")
    output = "probabilities" if mode == "classification" else "logits"
    manifest = {
        "task": {"mode": mode},
        "labels": ["red", "blue"],
        "preprocessing": {"timm_config": CONFIG},
        "inputs": {"images": {"layout": "NHWC"}},
        "tflite_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "tflite": {"outputs": {output: {"index": 7}}},
        "segmentation": {"background_logit_bias": bias},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class FakeInterpreter:
    def __init__(self, *args, **kwargs):
        self.input = None
        self.output = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 3}]

    def set_tensor(self, index, value):
        assert index == 3
        self.input = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        assert index == 7
        return self.output


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.interpreter = FakeInterpreter()
        fake_tf = types.SimpleNamespace(
            lite=types.SimpleNamespace(Interpreter=lambda **kwargs: self.interpreter)
        )
        self.fake_tf = fake_tf
        self.patch = patch.dict(sys.modules, {"tensorflow": fake_tf})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_classification_uses_manifest_output_index_and_nhwc(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_export(folder, "classification")
            self.interpreter.output = np.array([[0.8, 0.2]], dtype=np.float32)
            model = demo.Model(folder)
            crop, prediction = model.predict(Image.new("RGB", (6, 4), "red"))
            self.assertEqual(self.interpreter.input.shape, (1, 4, 4, 3))
            self.assertEqual(crop.size, (4, 4))
            self.assertEqual(demo.describe(prediction, model.labels), "red: 80.0%, blue: 20.0%")

    def test_segmentation_applies_saved_background_bias_once(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_export(folder, "segmentation", bias=0.0)
            logits = np.zeros((1, 3, 2, 2), dtype=np.float32)
            logits[:, 1:] = -2
            logits[0, 1, 0, 0] = 2
            logits[0, 2, 1, 0] = 2
            self.interpreter.output = logits
            model = demo.Model(folder / "manifest.json")
            crop, prediction = model.predict(Image.new("RGB", (6, 4), "white"))
            np.testing.assert_array_equal(prediction["mask"], [[0, -1], [1, -1]])
            self.assertEqual(demo.annotate(crop, prediction, model.labels).size, (4, 4))
            make_export(folder, "segmentation", bias=5.0)
            model = demo.Model(folder)
            _, prediction = model.predict(Image.new("RGB", (6, 4), "white"))
            np.testing.assert_array_equal(prediction["mask"], -np.ones((2, 2), dtype=np.int64))

    def test_bad_checksum_fails_before_loading_interpreter(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_export(folder, "classification")
            (folder / "model.tflite").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                demo.Model(folder)

    def test_only_known_tensorflow_interpreter_notice_is_suppressed(self):
        known = (
            "    Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in\n"
            "TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.\n"
            "See the migration guide for details."
        )
        unrelated = "An unrelated interpreter warning"

        def noisy_interpreter(**kwargs):
            warnings.warn_explicit(
                known,
                UserWarning,
                filename="interpreter.py",
                lineno=457,
                module="tensorflow.lite.python.interpreter",
            )
            warnings.warn_explicit(
                unrelated,
                UserWarning,
                filename="interpreter.py",
                lineno=458,
                module="tensorflow.lite.python.interpreter",
            )
            return self.interpreter

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_export(folder, "classification")
            with (
                patch.object(self.fake_tf.lite, "Interpreter", noisy_interpreter),
                warnings.catch_warnings(record=True) as observed,
            ):
                warnings.simplefilter("always")
                demo.Model(folder)
                # The temporary filter must not leak into the caller's context.
                warnings.warn_explicit(known, UserWarning, "interpreter.py", 459)
        self.assertEqual([str(item.message) for item in observed], [unrelated, known])

    def test_wrong_or_broken_manifest_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text("not JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                demo.Model(path)
            path.write_text('{"onnx_sha256": "test"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ONNX export"):
                demo.Model(path)
            path.write_text('{"tflite_sha256": "test", "task": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing its task or labels"):
                demo.Model(path)

    def test_input_cannot_be_its_own_results_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "photo.jpg").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "outside the input folder"):
                demo.image_files(folder, folder)

    def test_predict_preserves_old_result_and_fails_when_all_images_fail(self):
        class WorkingModel:
            mode = "classification"
            labels = ("red", "blue")

            def __init__(self, path):
                pass

            def predict(self, image):
                return image.copy(), {"mode": self.mode, "scores": np.array([0.8, 0.2])}

        class FailingModel(WorkingModel):
            def predict(self, image):
                raise ValueError("fixture failure")

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "photo.png"
            Image.new("RGB", (32, 32), "red").save(image)
            output = folder / "results"
            output.mkdir()
            old = output / "photo.png.png"
            old.write_bytes(b"original")
            with patch.object(predict, "Model", WorkingModel):
                predict.main(["unused-export", str(image), "--output", str(output)])
            self.assertEqual(old.read_bytes(), b"original")
            self.assertTrue((output / "photo.png_2.png").is_file())
            with (
                patch.object(predict, "Model", FailingModel),
                self.assertRaisesRegex(SystemExit, "No images were processed"),
            ):
                predict.main(["unused-export", str(image), "--output", str(output)])

    def test_webcam_keeps_reading_while_predicting_and_stops_on_window_close(self):
        second_frame = threading.Event()
        prediction_timed_out = []

        class Camera:
            count = 0
            released = False

            def isOpened(self):
                return True

            def read(self):
                self.count += 1
                if self.count >= 2:
                    second_frame.set()
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        class SlowModel:
            config = CONFIG
            labels = ("red", "blue")

            def __init__(self, path):
                pass

            def predict(self, image):
                if not second_frame.wait(timeout=1):
                    prediction_timed_out.append(True)
                crop = demo.prepare_image(image, self.config)[1]
                return crop, {"mode": "classification", "scores": np.array([0.8, 0.2])}

        camera = Camera()
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda index: camera,
            COLOR_BGR2RGB=1,
            COLOR_RGB2BGR=2,
            FONT_HERSHEY_SIMPLEX=0,
            WND_PROP_VISIBLE=0,
            error=RuntimeError,
            cvtColor=lambda frame, code: frame,
            putText=lambda *args: None,
            imshow=lambda *args: None,
            waitKey=lambda *args: 0,
            getWindowProperty=lambda *args: 0 if camera.count >= 3 else 1,
            destroyAllWindows=lambda: None,
        )
        with patch.dict(sys.modules, {"demo": demo, "cv2": fake_cv2}):
            webcam_spec = importlib.util.spec_from_file_location(
                "ijk_tflite_webcam", ROOT / "webcam.py"
            )
            webcam = importlib.util.module_from_spec(webcam_spec)
            webcam_spec.loader.exec_module(webcam)
        with patch.object(webcam, "Model", SlowModel):
            webcam.main(["unused-export", "--every", "1"])
        self.assertGreaterEqual(camera.count, 3)
        self.assertTrue(camera.released)
        self.assertFalse(prediction_timed_out)

    def test_webcam_default_samples_the_fifth_frame(self):
        analyzed = []

        class Camera:
            count = 0

            def isOpened(self):
                return True

            def read(self):
                self.count += 1
                return True, np.full((8, 8, 3), self.count, dtype=np.uint8)

            def release(self):
                pass

        class Model:
            config = CONFIG
            labels = ("red", "blue")

            def __init__(self, path):
                pass

            def predict(self, image):
                analyzed.append(int(np.asarray(image)[0, 0, 0]))
                crop = demo.prepare_image(image, self.config)[1]
                return crop, {"mode": "classification", "scores": np.array([0.8, 0.2])}

        camera = Camera()
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda index: camera,
            COLOR_BGR2RGB=1,
            COLOR_RGB2BGR=2,
            FONT_HERSHEY_SIMPLEX=0,
            WND_PROP_VISIBLE=0,
            error=RuntimeError,
            cvtColor=lambda frame, code: frame,
            putText=lambda *args: None,
            imshow=lambda *args: None,
            waitKey=lambda *args: 0,
            getWindowProperty=lambda *args: 0 if camera.count >= 6 else 1,
            destroyAllWindows=lambda: None,
        )
        with patch.dict(sys.modules, {"demo": demo, "cv2": fake_cv2}):
            webcam_spec = importlib.util.spec_from_file_location(
                "ijk_tflite_webcam_fifth", ROOT / "webcam.py"
            )
            webcam = importlib.util.module_from_spec(webcam_spec)
            webcam_spec.loader.exec_module(webcam)
        with patch.object(webcam, "Model", Model):
            webcam.main(["unused-export"])
        self.assertEqual(analyzed, [5])


if __name__ == "__main__":
    unittest.main()
