from .autoencoder import Autoencoder
from .base import AnomalyModel
from .feature_ae import FeatureAE, feature_ae_res2net50, feature_ae_resnet50
from .knn import PatchKNN, ResNetKNN, patchcore

MODELS = {
    "autoencoder_l2": Autoencoder,
    "resnet50_knn": ResNetKNN,
    "feature_ae_resnet50": feature_ae_resnet50,
    "feature_ae_res2net50": feature_ae_res2net50,
    "patchcore": patchcore,
}

__all__ = ["MODELS", "AnomalyModel", "Autoencoder", "FeatureAE", "PatchKNN", "ResNetKNN", "patchcore",
           "feature_ae_resnet50", "feature_ae_res2net50"]
