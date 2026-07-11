from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.dataset import discover_medical_samples


def test_discover_medical_samples_infers_labels_from_folder_names(tmp_path):
    benign_dir = tmp_path / "benign"
    malignant_dir = tmp_path / "malignant"
    benign_dir.mkdir(parents=True)
    malignant_dir.mkdir(parents=True)

    benign_image = np.zeros((64, 64), dtype=np.float32)
    malignant_image = np.ones((64, 64), dtype=np.float32)
    np.save(benign_dir / "sample_0.npy", benign_image)
    np.save(malignant_dir / "sample_1.npy", malignant_image)

    manifest = discover_medical_samples(tmp_path)

    assert len(manifest) == 2
    assert {item["label"] for item in manifest} == {0, 1}
    assert any(item["label_name"] == "benign" for item in manifest)
    assert any(item["label_name"] == "malignant" for item in manifest)


def test_predictor_prepares_ct_like_images_for_model_input(tmp_path):
    image_path = tmp_path / "ct_slice.png"
    ct_slice = np.random.randint(-1000, 400, size=(64, 64), dtype=np.int16)
    Image.fromarray(ct_slice.astype(np.uint8)).save(image_path)

    from backend.ml.predictor import LungCancerPredictor

    predictor = LungCancerPredictor(model_path=None)
    image, tensor = predictor._prepare_tensor(image_path.read_bytes())

    assert image is not None
    assert tensor.shape[1:] == (3, 512, 512)
