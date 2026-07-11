import torch.nn as nn

from backend.ml.predictor import LungCancerPredictor


def test_predictor_uses_pretrained_backbone_by_default(monkeypatch):
    captured = {}

    class DummyClassifier(nn.Module):
        def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
            super().__init__()
            captured["pretrained"] = pretrained
            self.backbone = nn.Identity()
            self.num_classes = num_classes

    monkeypatch.setattr("backend.ml.predictor.DenseNetClassifier", DummyClassifier)

    predictor = LungCancerPredictor(model_path=None)

    assert predictor.model is not None
    assert captured["pretrained"] is True
