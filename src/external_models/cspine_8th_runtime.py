from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from importlib import util
from pathlib import Path

import numpy as np
import pydicom
import torch

from .. import config


SOLUTION_NAME = "8th-place-solution"
KAGGLE_MODEL_URL = "https://www.kaggle.com/models/zixuanh/cspine-8th-place-solution-model-weights"
OUTPUT_ROOT = config.DEMO_DIR / "outputs" / "cspine_reference_probe"
REPO_DIR = OUTPUT_ROOT / "repo" / SOLUTION_NAME
WEIGHTS_DIR = OUTPUT_ROOT / "weights" / SOLUTION_NAME

REQUIRED_WEIGHT_DIRS = (
    "try2-seg-b1v10-sagview-full",
    "try2-seg-b1v1-full",
    "try17-b5-v5-t4-pseudo-round1",
    "b5-v5-t4-pseudo-round1-seq-v2",
)

VENDORED_PATHS = (
    REPO_DIR / "timm-pytorch-image-models" / "pytorch-image-models-master",
    REPO_DIR / "segmentation-models-pytorch" / "segmentation_models.pytorch-master",
    REPO_DIR / "efficientnet-pytorch" / "EfficientNet-PyTorch-master",
    REPO_DIR / "pretrainedmodels" / "pretrainedmodels-0.7.4" / "pretrainedmodels-0.7.4",
)

REQUIRED_IMPORTS = (
    "torch",
    "cv2",
    "timm",
    "segmentation_models_pytorch",
    "efficientnet_pytorch",
    "pretrainedmodels",
)

TARGETS = ("patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7")
FALLBACK_MEANS = np.array([0.4760, 0.0723, 0.1412, 0.0362, 0.0535, 0.0802, 0.1372, 0.1947], dtype=np.float32)
_MODEL_CACHE: dict[tuple[str, int], "CSpine8thModelBundle"] = {}


@dataclass(frozen=True)
class CSpine8thReadiness:
    repo_exists: bool
    notebook_exists: bool
    weights_dir: Path
    missing_weight_dirs: tuple[str, ...]
    missing_imports_before_vendor: tuple[str, ...]
    missing_imports_after_vendor: tuple[str, ...]
    added_vendor_paths: tuple[str, ...]

    @property
    def runtime_ready(self) -> bool:
        return (
            self.repo_exists
            and self.notebook_exists
            and not self.missing_weight_dirs
            and not self.missing_imports_after_vendor
        )

    def to_dict(self) -> dict:
        return {
            "repo_exists": self.repo_exists,
            "notebook_exists": self.notebook_exists,
            "weights_dir": str(self.weights_dir),
            "missing_weight_dirs": list(self.missing_weight_dirs),
            "missing_imports_before_vendor": list(self.missing_imports_before_vendor),
            "missing_imports_after_vendor": list(self.missing_imports_after_vendor),
            "added_vendor_paths": list(self.added_vendor_paths),
            "runtime_ready": self.runtime_ready,
        }


def add_vendor_paths() -> tuple[str, ...]:
    added = []
    for path in VENDORED_PATHS:
        if path.exists():
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
                added.append(path_text)
    return tuple(added)


def check_imports() -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_IMPORTS if util.find_spec(name) is None)


def check_readiness(add_vendor: bool = True) -> CSpine8thReadiness:
    missing_before = check_imports()
    added_paths = add_vendor_paths() if add_vendor else ()
    missing_after = check_imports()
    missing_weight_dirs = tuple(name for name in REQUIRED_WEIGHT_DIRS if not (WEIGHTS_DIR / name).exists())
    return CSpine8thReadiness(
        repo_exists=REPO_DIR.exists(),
        notebook_exists=(REPO_DIR / "8th-place-inference.ipynb").exists(),
        weights_dir=WEIGHTS_DIR,
        missing_weight_dirs=missing_weight_dirs,
        missing_imports_before_vendor=missing_before,
        missing_imports_after_vendor=missing_after,
        added_vendor_paths=added_paths,
    )


def format_readiness(readiness: CSpine8thReadiness) -> str:
    lines = [
        "RSNA C-Spine 8th Reference",
        "用途：跨域颈椎骨折参考模型，不是当前手腕/足部骨折诊断模型。",
        f"Kaggle weights: {KAGGLE_MODEL_URL}",
        f"Expected weights dir: {readiness.weights_dir}",
        f"Repo exists: {readiness.repo_exists}",
        f"Notebook exists: {readiness.notebook_exists}",
        f"Runtime ready: {readiness.runtime_ready}",
    ]
    if readiness.added_vendor_paths:
        lines.append("Vendored paths added:")
        lines.extend(f"  - {path}" for path in readiness.added_vendor_paths)
    if readiness.missing_imports_after_vendor:
        lines.append("Missing imports:")
        lines.extend(f"  - {name}" for name in readiness.missing_imports_after_vendor)
    if readiness.missing_weight_dirs:
        lines.append("Missing weight folders:")
        lines.extend(f"  - {readiness.weights_dir / name}" for name in readiness.missing_weight_dirs)
    if readiness.runtime_ready:
        lines.append("Current status: runtime prerequisites are ready; use Run C-Spine Reference to execute the extracted wrapper.")
    else:
        lines.append("Current status: not runnable yet; waiting for weights and/or dependency resolution.")
    return "\n".join(lines)


def run_cspine_reference_case(case_path: str) -> dict:
    readiness = check_readiness()
    if not readiness.runtime_ready:
        raise RuntimeError(format_readiness(readiness))
    started = time.time()
    device = "cuda" if _torch().cuda.is_available() else "cpu"
    max_folds = 1
    bundle = get_model_bundle(device=device, max_folds=max_folds)
    probabilities = bundle.predict_case(Path(case_path))
    elapsed = round(time.time() - started, 2)
    c1_c7 = {f"C{idx}": float(probabilities[idx]) for idx in range(1, 8)}
    return {
        "model": "RSNA C-Spine 8th Reference",
        "out_of_domain": True,
        "device": device,
        "max_folds_per_stage": max_folds,
        "elapsed_seconds": elapsed,
        "fallback_used": bundle.last_fallback_reason is not None,
        "fallback_reason": bundle.last_fallback_reason,
        "fallback_traceback": bundle.last_fallback_traceback,
        "study_probability": float(probabilities[0]),
        "c1_c7_probabilities": c1_c7,
        "note": "Cervical-spine RSNA 2022 reference model; not trained for wrist/foot fracture detection.",
    }


def get_model_bundle(device: str, max_folds: int) -> "CSpine8thModelBundle":
    key = (device, max_folds)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = CSpine8thModelBundle(device=device, max_folds=max_folds)
    return _MODEL_CACHE[key]


class CSpine8thModelBundle:
    def __init__(self, device: str, max_folds: int = 1) -> None:
        add_vendor_paths()
        self.device = device
        self.max_folds = max(1, max_folds)
        self.torch = _torch()
        self.nn = self.torch.nn
        self.sag_models = self._load_models(SagSegModel, WEIGHTS_DIR / "try2-seg-b1v10-sagview-full", state_dict_key="state_dict")
        self.bone_models = self._load_models(BoneSegModel, WEIGHTS_DIR / "try2-seg-b1v1-full", state_dict_key="state_dict")
        self.cls_models = self._load_models(SliceClsModel, WEIGHTS_DIR / "try17-b5-v5-t4-pseudo-round1")
        self.seq_models = self._load_models(lambda: SeqModel(seq_dim=64), WEIGHTS_DIR / "b5-v5-t4-pseudo-round1-seq-v2")
        self.last_fallback_reason: str | None = None
        self.last_fallback_traceback: str | None = None

    def _load_models(self, factory, folder: Path, state_dict_key: str | None = None) -> list:
        models = []
        files = [
            path
            for path in sorted(folder.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.lower() in {".ckpt", ".pth", ".pt"} and not path.name.startswith("._")
        ]
        for path in files[: self.max_folds]:
            model = factory()
            model.eval()
            model.to(self.device)
            state = self.torch.load(str(path), map_location=self.device)
            if state_dict_key is not None:
                state = state[state_dict_key]
            model.load_state_dict(state, strict=(state_dict_key is None))
            models.append(model)
        if not models:
            raise FileNotFoundError(f"No model weights found in {folder}")
        return models

    def predict_case(self, case_path: Path) -> np.ndarray:
        torch = self.torch
        nn = self.nn
        self.last_fallback_reason = None
        self.last_fallback_traceback = None
        try:
            images = load_case_images(case_path)
            if images.shape[0] < 5:
                raise RuntimeError(f"Too few slices for C-Spine model: {images.shape[0]}")

            sag = images[:, :, images.shape[-1] // 2]
            keys = np.array(sag_inference(self.sag_models, sag, images.shape[0], self.device), dtype=np.int64)
            selected = np.where(np.logical_and(keys != 100, keys != 8))[0]
            images = images[selected]
            keys = keys[selected]
            if len(images) < 3:
                raise RuntimeError("Sagittal model did not select enough cervical slices.")

            masks = bone_inference(self.bone_models, images, size=256, batch_size=32, device=self.device)
            masks = np.max(masks, axis=-1)
            selected = [idx for idx, mask in enumerate(masks) if np.max(mask)]
            masks = masks[selected]
            images = images[selected]
            keys = keys[selected]
            if len(images) < 3:
                raise RuntimeError("Bone segmentation did not retain enough slices.")

            inputs = []
            for idx in range(1, len(images) - 1):
                image = np.stack([images[idx - 1], images[idx], images[idx + 1]], axis=-1)
                image = crop_to_mask_roi(image, masks[idx])
                inputs.append(preprocess_cls_image(image))
            if not inputs:
                raise RuntimeError("No 2.5D classification inputs were produced.")

            cls_images = torch.stack(inputs)
            keys = keys[1:-1]
            cls_images = nn.functional.interpolate(cls_images, (456, 456))
            _, features = cls_inference(self.cls_models, cls_images, batch_size=32, device=self.device)

            bone_features = []
            seq_dim = 64
            for bone in range(1, 8):
                feature_block = np.zeros((seq_dim, 2048), dtype=np.float32)
                if np.sum(keys == bone):
                    bone_slice_features = features[keys == bone]
                    feature_block[: min(len(bone_slice_features), seq_dim)] = bone_slice_features[:seq_dim]
                bone_features.append(feature_block)

            bone_features_tensor = torch.as_tensor(np.stack(bone_features)).float().to(self.device)
            seq_outputs = []
            with torch.no_grad():
                for model in self.seq_models:
                    output = model.sigmoid(model(bone_features_tensor)[0]).detach().cpu().numpy()
                    seq_outputs.append(output)
            probabilities = np.mean(seq_outputs, axis=0).astype(np.float32)
            return np.clip(probabilities, 0.001, 0.999)
        except Exception as exc:
            import traceback

            self.last_fallback_reason = f"{type(exc).__name__}: {exc}"
            self.last_fallback_traceback = traceback.format_exc()
            return FALLBACK_MEANS.copy()
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()


class SagSegModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        smp = _segmentation_models_pytorch()
        torch = _torch()
        self.feature_extractor = smp.Unet("tu-tf_efficientnet_b1_ns", in_channels=1, classes=8, encoder_weights=None)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp):
        return self.feature_extractor(inp)


class BoneSegModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        smp = _segmentation_models_pytorch()
        torch = _torch()
        self.feature_extractor = smp.Unet("tu-tf_efficientnet_b1_ns", in_channels=3, classes=8, encoder_weights=None)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp):
        return self.feature_extractor(inp)


class SliceClsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch = _torch()
        timm = _timm()
        self.feature_extractor = timm.models.tf_efficientnet_b5_ns(in_chans=3, pretrained=False, num_classes=0, global_pool="")
        features = self.feature_extractor.num_features
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(features, 1)
        self.flatten = torch.nn.Flatten()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp):
        features = self.feature_extractor(inp)
        features = self.avgpool(features)
        features = self.flatten(features)
        logits = self.classifier(features)
        return logits, features


class Attention(torch.nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super().__init__()
        torch = _torch()
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        weight = torch.zeros(feature_dim, 1)
        torch.nn.init.xavier_uniform_(weight)
        self.weight = torch.nn.Parameter(weight)
        if bias:
            self.b = torch.nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        torch = _torch()
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.weight).view(-1, self.step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        return torch.sum(x * torch.unsqueeze(a, -1), 1)


class SeqModel(torch.nn.Module):
    def __init__(self, seq_dim=64):
        super().__init__()
        torch = _torch()
        nn = torch.nn
        base = 2048
        self.lstm1 = nn.GRU(base, 512, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(1024, 512, bidirectional=True, batch_first=True)
        self.attention1 = Attention(1024, seq_dim)
        self.conv1 = nn.Conv1d(seq_dim, 1, 1)
        self.lstm_bn1 = nn.BatchNorm1d(seq_dim)
        self.lstm_bn2 = nn.BatchNorm1d(seq_dim)
        self.att_bn1 = nn.BatchNorm1d(1024)
        self.conv_bn1 = nn.BatchNorm1d(1)
        self.clf = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(0.2)
        self.final_classifier = nn.Linear(2048 * 7, 8)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, inp):
        torch = _torch()
        x, _ = self.lstm1(inp)
        x = self.tanh(x)
        x = self.lstm_bn1(x)
        x, _ = self.lstm2(x)
        x = self.tanh(x)
        x = self.lstm_bn2(x)
        x = self.relu(x)
        x_conv = self.conv1(x)[:, 0]
        x = self.attention1(x, mask=None)
        x = self.att_bn1(x)
        x = self.relu(x)
        x = torch.cat([x, x_conv], -1)
        x = self.dropout(x)
        features = x.reshape(x.shape[0] // 7, 7, 2048)
        features = torch.nn.Flatten(1, 2)(features)
        return self.final_classifier(features)


def load_case_images(case_path: Path) -> np.ndarray:
    dicoms = []
    for path in case_path.rglob("*"):
        if not path.is_file() or path.name.upper() in {"DICOMDIR", "LOCKFILE", "VERSION"}:
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if getattr(ds, "Rows", None) and getattr(ds, "Columns", None):
                dicoms.append((path, ds))
        except Exception:
            continue
    if not dicoms:
        raise FileNotFoundError(f"No DICOM slices found under {case_path}")
    sorted_items = sorted(dicoms, key=lambda item: dicom_z_position(item[1]), reverse=True)
    images = [preprocess_dicom_pixels(ds) for _, ds in sorted_items]
    return np.stack(images)


def dicom_z_position(ds) -> float:
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[-1])
    return float(getattr(ds, "InstanceNumber", 0))


def preprocess_dicom_pixels(ds) -> np.ndarray:
    image = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.max(image) - image
    min_value = float(np.min(image))
    if min_value > 0:
        image = image - min_value
    elif min_value < 0:
        image = image + abs(min_value)
    max_value = float(np.max(image))
    if max_value <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    image = image / max_value
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def sag_inference(models, image: np.ndarray, length: int, device: str) -> list[int]:
    torch = _torch()
    cv2 = _cv2()
    with torch.no_grad():
        max_value = max(float(np.max(image)), 1.0)
        img = cv2.resize(image, (256, 256)).astype(np.float32) / max_value
        outputs = []
        for model in models:
            tensor = torch.as_tensor(img).unsqueeze(0).unsqueeze(0).to(device)
            output = model.sigmoid(model(tensor))[0].detach().cpu().numpy().transpose(1, 2, 0)
            output = cv2.resize(output, (image.shape[1], length))
            outputs.append(output)
        output = np.mean(outputs, axis=0)
        output = (output > 0.3).astype(np.uint8)
        preds = []
        for row in output:
            classes = np.sum(row, axis=0)
            preds.append(int(np.argmax(classes) + 1) if np.any(classes) else 100)
        return preds


def bone_inference(models, images: np.ndarray, size: int, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    nn = torch.nn
    with torch.no_grad():
        tensor = nn.functional.interpolate(torch.as_tensor(images).unsqueeze(1), (size, size))
        tensor = torch.cat([tensor] * 3, dim=1)
        outputs = []
        for start in range(0, tensor.shape[0], batch_size):
            batch_outputs = []
            inputs = tensor[start : start + batch_size].to(device).float() / 255
            for model in models:
                output = model.sigmoid(model(inputs)).detach().cpu().numpy().transpose(0, 2, 3, 1)
                batch_outputs.append(output)
            output = np.mean(np.stack(batch_outputs), axis=0)
            outputs.extend((output > 0.5).astype(np.uint8))
        return np.stack(outputs)


def crop_to_mask_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return image
    ymin = max(float(np.min(ys)) / mask.shape[0] * 0.95, 0.0)
    ymax = min(float(np.max(ys)) / mask.shape[0] * 1.05, 1.0)
    xmin = max(float(np.min(xs)) / mask.shape[1] * 0.95, 0.0)
    xmax = min(float(np.max(xs)) / mask.shape[1] * 1.05, 1.0)
    y0, y1 = int(ymin * image.shape[0]), int(ymax * image.shape[0])
    x0, x1 = int(xmin * image.shape[1]), int(xmax * image.shape[1])
    if y1 <= y0 or x1 <= x0:
        return image
    return image[y0:y1, x0:x1]


def preprocess_cls_image(image: np.ndarray):
    torch = _torch()
    cv2 = _cv2()
    height, width = image.shape[:2]
    scale = 1024 / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_top = (1024 - new_height) // 2
    pad_bottom = 1024 - new_height - pad_top
    pad_left = (1024 - new_width) // 2
    pad_right = 1024 - new_width - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return torch.from_numpy(padded.transpose(2, 0, 1).copy())


def cls_inference(models, images, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    with torch.no_grad():
        outputs = []
        features = []
        for start in range(0, images.shape[0], batch_size):
            inputs = images[start : start + batch_size].to(device).float() / 255
            batch_outputs = []
            batch_features = []
            for model in models:
                output, feature = model(inputs)
                batch_outputs.append(model.sigmoid(output).detach().cpu().numpy())
                batch_features.append(feature.detach().cpu().numpy())
            outputs.extend(np.mean(np.stack(batch_outputs), axis=0))
            features.extend(np.mean(np.stack(batch_features), axis=0))
        return np.stack(outputs), np.stack(features)


def _torch():
    return torch


def _cv2():
    import cv2

    return cv2


def _timm():
    add_vendor_paths()
    import timm

    return timm


def _segmentation_models_pytorch():
    add_vendor_paths()
    import segmentation_models_pytorch as smp

    return smp
