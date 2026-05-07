from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from app.core.config import Settings
from app.models.neural_networks import BinaryUNetSmall, LastVisibleEstimator, UNetEnhanced

CLASS_NAMES = [
    "background",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
]
CANONICAL_LABELS = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
LABEL_TO_CLASS_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE_BINARY = (512, 256)
IMG_SIZE_MULTICLASS = (640, 320)
IMG_SIZE_LAST = (384, 192)
BINARY_THRESHOLD = 0.50
ROI_PAD_X = 28
ROI_PAD_Y = 44
MIN_FOREGROUND_PIXELS = 24
PROFILE_BINS = 24
PRESENCE_THRESHOLD_PIXELS = 40
ASSUMED_FIRST_VISIBLE_IDX = 0
USE_MULTICLASS_TTA = True
LAST_EXPECTATION_BLEND = 0.30
LAST_HEURISTIC_BLEND = 0.20
MIN_COMPONENT_PIXELS = 48
SECONDARY_COMPONENT_RATIO = 0.35
MAX_LABEL_GAP_FOR_MAIN_BAND = 2
ALLOW_MARGIN_LABELS = 1
MIN_AREA_RATIO_TO_KEEP_EDGE_OUTLIER = 0.45
MONOTONIC_TOLERANCE_PX = 6.0
RESTORE_EDGE_MIN_AREA_RATIO = 0.30
RESTORE_NEIGHBOR_DISTANCE = 1

COLOR_MAP = np.array(
    [
        [0, 0, 0],
        [46, 125, 179],
        [67, 153, 118],
        [244, 170, 66],
        [218, 89, 71],
        [120, 99, 178],
        [64, 167, 185],
        [236, 121, 176],
        [134, 176, 73],
        [185, 126, 58],
        [104, 137, 208],
        [205, 98, 130],
        [90, 160, 130],
        [242, 190, 77],
        [160, 108, 192],
        [81, 145, 203],
        [224, 112, 91],
        [93, 179, 151],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class VertebraGeometry:
    label: str
    mask_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_pixels: int
    orientation_degrees: float | None


@dataclass(frozen=True)
class PipelineResult:
    predicted_labels: list[str]
    vertebrae: list[VertebraGeometry]
    mask_path: Path
    preview_path: Path
    bbox: tuple[int, int, int, int]
    pred_last_label: str


def read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def resize_image(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size[1], size[0]), resample=Image.BILINEAR))


def resize_mask(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize((size[1], size[0]), resample=Image.NEAREST))


def bbox_from_mask(mask: np.ndarray, min_foreground_pixels: int = 24) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_foreground_pixels:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clamp_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad_x: int = 28,
    pad_y: int = 44,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return clamp_bbox((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), image_shape)


def scale_bbox(
    bbox: tuple[int, int, int, int],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x0, y0, x1, y1 = bbox
    sx = dst_w / src_w
    sy = dst_h / src_h
    return clamp_bbox(
        (int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))),
        dst_shape,
    )


def crop_array(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return arr[y0:y1, x0:x1]


def normalize_image(image_2d: np.ndarray) -> np.ndarray:
    mean = float(image_2d.mean())
    std = float(image_2d.std())
    if std < 1e-6:
        return image_2d - mean
    return (image_2d - mean) / std


def build_coordinate_channels(height: int, width: int) -> np.ndarray:
    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y_map = np.repeat(y_coords, width, axis=1)
    x_map = np.repeat(x_coords, height, axis=0)
    return np.stack([y_map, x_map], axis=0)


def class_stats_from_mask(mask_2d: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    total_fg = float((mask_2d > 0).sum()) + 1e-6
    for class_id in range(1, NUM_CLASSES):
        class_mask = mask_2d == class_id
        if not class_mask.any():
            continue
        ys, _ = np.where(class_mask)
        class_name = CLASS_NAMES[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "label_index": CANONICAL_LABELS.index(class_name),
                "area_pixels": int(class_mask.sum()),
                "area_ratio": float(class_mask.sum() / total_fg),
                "centroid_y": float(np.mean(ys)),
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
            }
        )
    return sorted(rows, key=lambda row: int(row["label_index"]))


def build_label_blocks(label_indices: list[int], gap_tolerance: int = 2) -> list[list[int]]:
    if not label_indices:
        return []
    blocks = [[label_indices[0]]]
    for idx in label_indices[1:]:
        if idx - blocks[-1][-1] <= gap_tolerance + 1:
            blocks[-1].append(idx)
        else:
            blocks.append([idx])
    return blocks


def select_dominant_label_band(stats: list[dict[str, float | int | str]]) -> tuple[int | None, int | None]:
    if not stats:
        return None, None
    indices = [int(row["label_index"]) for row in stats]
    blocks = build_label_blocks(indices, gap_tolerance=MAX_LABEL_GAP_FOR_MAIN_BAND)
    scored = []
    for block in blocks:
        area = sum(float(row["area_pixels"]) for row in stats if int(row["label_index"]) in block)
        scored.append(((len(block), area, -float(min(block))), block[0], block[-1]))
    scored = sorted(scored, reverse=True)
    return int(scored[0][1]), int(scored[0][2])


class ScoliosisPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aux_dim = 146
        self.aux_mean = np.zeros(self.aux_dim, dtype=np.float32)
        self.aux_std = np.ones(self.aux_dim, dtype=np.float32)
        self.binary_model: BinaryUNetSmall | None = None
        self.multiclass_model: UNetEnhanced | None = None
        self.last_model: LastVisibleEstimator | None = None

    def load(self) -> None:
        if self.binary_model is not None:
            return

        binary_model = BinaryUNetSmall(in_channels=1, out_channels=1).to(self.device)
        binary_model.load_state_dict(
            torch.load(
                self.settings.model_dir / self.settings.binary_model_name,
                map_location=self.device,
                weights_only=True,
            )
        )
        binary_model.eval()

        multiclass_model = UNetEnhanced(in_channels=3, out_channels=NUM_CLASSES, base=48, dropout=0.10).to(self.device)
        multiclass_model.load_state_dict(
            torch.load(
                self.settings.model_dir / self.settings.multiclass_model_name,
                map_location=self.device,
                weights_only=True,
            )
        )
        multiclass_model.eval()

        last_model = LastVisibleEstimator(aux_dim=self.aux_dim, num_labels=len(CANONICAL_LABELS), dropout=0.25).to(self.device)
        last_model.load_state_dict(
            torch.load(
                self.settings.model_dir / self.settings.last_visible_model_name,
                map_location=self.device,
                weights_only=True,
            )
        )
        last_model.eval()

        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.last_model = last_model

    def predict(self, image_path: Path, prediction_id: str) -> PipelineResult:
        self.load()
        image_raw = read_gray(image_path)
        image_shape = image_raw.shape

        bbox_small = self._predict_binary_bbox_from_image(image_raw)
        if bbox_small is None:
            bbox = (0, 0, image_shape[1], image_shape[0])
        else:
            bbox = scale_bbox(bbox_small, src_shape=IMG_SIZE_BINARY, dst_shape=image_shape)
            bbox = expand_bbox(bbox, image_shape=image_shape, pad_x=ROI_PAD_X, pad_y=ROI_PAD_Y)

        image_crop, raw_mask, prob_map = self._infer_multiclass_on_bbox(image_raw, bbox)
        post_mask = self._postprocess_mask(raw_mask)
        aux_features = self._extract_aux_features_from_prediction(post_mask, prob_map)
        aux_norm = ((aux_features - self.aux_mean) / self.aux_std).astype(np.float32)
        heuristic_last_idx = self._estimate_last_visible_from_mask(post_mask)

        coords_last = build_coordinate_channels(IMG_SIZE_LAST[0], IMG_SIZE_LAST[1])
        image_small = resize_image((image_crop * 255.0).astype(np.uint8), IMG_SIZE_LAST).astype(np.float32) / 255.0
        image_small = normalize_image(image_small)
        image_last = np.concatenate([np.expand_dims(image_small, axis=0), coords_last], axis=0)

        image_tensor = torch.tensor(image_last[None, ...], dtype=torch.float32, device=self.device)
        aux_tensor = torch.tensor(aux_norm[None, ...], dtype=torch.float32, device=self.device)
        assert self.last_model is not None
        with torch.no_grad():
            logits = self.last_model(image_tensor, aux_tensor)
        pred_last_idx = self._blend_last_prediction(logits, heuristic_last_idx)
        final_mask_roi = self._clip_mask_to_last_idx(post_mask, ASSUMED_FIRST_VISIBLE_IDX, pred_last_idx)

        full_mask = self._project_roi_mask_to_full_image(final_mask_roi, bbox, image_shape)
        labels_final = self._present_labels_from_mask(full_mask)
        vertebrae = self._extract_vertebrae_geometry(full_mask)
        pred_last_label = CANONICAL_LABELS[pred_last_idx]

        mask_path = self.settings.results_dir / f"{prediction_id}_mask.png"
        preview_path = self.settings.results_dir / f"{prediction_id}_preview.png"
        Image.fromarray(full_mask.astype(np.uint8)).save(mask_path)
        self._save_preview(image_raw, full_mask, preview_path, labels_final, pred_last_label)

        return PipelineResult(
            predicted_labels=labels_final,
            vertebrae=vertebrae,
            mask_path=mask_path,
            preview_path=preview_path,
            bbox=bbox,
            pred_last_label=pred_last_label,
        )

    def _predict_binary_bbox_from_image(self, image_raw: np.ndarray) -> tuple[int, int, int, int] | None:
        assert self.binary_model is not None
        image_resized = resize_image(image_raw, IMG_SIZE_BINARY).astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_resized[None, None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.binary_model(image_tensor)
            pred_mask_small = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= BINARY_THRESHOLD).astype(np.uint8)
        return bbox_from_mask(pred_mask_small, min_foreground_pixels=MIN_FOREGROUND_PIXELS)

    def _infer_multiclass_on_bbox(self, image_raw: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.multiclass_model is not None
        image_crop = crop_array(image_raw, bbox)
        image_crop = resize_image(image_crop, IMG_SIZE_MULTICLASS).astype(np.float32) / 255.0
        image_crop = normalize_image(image_crop)
        coords = build_coordinate_channels(IMG_SIZE_MULTICLASS[0], IMG_SIZE_MULTICLASS[1])
        image_channels = np.concatenate([np.expand_dims(image_crop, axis=0), coords], axis=0)
        image_tensor = torch.tensor(image_channels[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.multiclass_model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            if USE_MULTICLASS_TTA:
                flipped_input = torch.flip(image_tensor, dims=[3])
                flipped_logits = self.multiclass_model(flipped_input)
                flipped_probs = torch.softmax(flipped_logits, dim=1)
                flipped_probs = torch.flip(flipped_probs, dims=[3])
                probs = 0.5 * (probs + flipped_probs)
            probs_np = probs[0].detach().cpu().numpy().astype(np.float32)
            pred_mask = np.argmax(probs_np, axis=0).astype(np.int64)
        return image_crop, pred_mask, probs_np

    def _extract_aux_features_from_prediction(self, pred_mask: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
        h, _ = pred_mask.shape
        fg_mask = (pred_mask > 0).astype(np.float32)
        total_fg = float(fg_mask.sum()) + 1e-6
        presence: list[float] = []
        area_ratio: list[float] = []
        centroid_y: list[float] = []
        y_min_norm: list[float] = []
        y_max_norm: list[float] = []
        height_span_norm: list[float] = []
        mean_confidence: list[float] = []

        for label in CANONICAL_LABELS:
            class_id = LABEL_TO_CLASS_ID[label]
            class_mask = pred_mask == class_id
            area = float(class_mask.sum())
            presence.append(1.0 if area >= PRESENCE_THRESHOLD_PIXELS else 0.0)
            area_ratio.append(area / total_fg)
            if area > 0:
                ys, _ = np.where(class_mask)
                centroid_y.append(float(np.mean(ys) / max(h - 1, 1)))
                y_min_norm.append(float(np.min(ys) / max(h - 1, 1)))
                y_max_norm.append(float(np.max(ys) / max(h - 1, 1)))
                height_span_norm.append(float((np.max(ys) - np.min(ys) + 1) / max(h, 1)))
                mean_confidence.append(float(prob_map[class_id][class_mask].mean()))
            else:
                centroid_y.append(0.0)
                y_min_norm.append(0.0)
                y_max_norm.append(0.0)
                height_span_norm.append(0.0)
                mean_confidence.append(0.0)

        pred_present_indices = [i for i, value in enumerate(presence) if value > 0.5]
        min_present_idx = float(min(pred_present_indices)) if pred_present_indices else 0.0
        max_present_idx = float(max(pred_present_indices)) if pred_present_indices else 0.0
        num_present = float(len(pred_present_indices))
        row_profile = fg_mask.sum(axis=1).astype(np.float32)
        if row_profile.max() > 0:
            row_profile = row_profile / row_profile.max()
        profile_features = [float(chunk.mean()) for chunk in np.array_split(row_profile, PROFILE_BINS)]

        return np.array(
            presence
            + area_ratio
            + centroid_y
            + y_min_norm
            + y_max_norm
            + height_span_norm
            + mean_confidence
            + [
                min_present_idx / (len(CANONICAL_LABELS) - 1),
                max_present_idx / (len(CANONICAL_LABELS) - 1),
                num_present / len(CANONICAL_LABELS),
            ]
            + profile_features,
            dtype=np.float32,
        )

    def _postprocess_mask(self, mask_2d: np.ndarray) -> np.ndarray:
        cleaned = self._keep_reliable_components_per_class(mask_2d)
        stats_initial = class_stats_from_mask(cleaned)
        start_idx, end_idx = select_dominant_label_band(stats_initial)
        trimmed = self._trim_edge_outliers(cleaned, stats_initial, start_idx, end_idx)
        monotonic = self._enforce_monotonic_centroids(trimmed)
        return self._restore_supported_edge_labels(monotonic, cleaned, start_idx, end_idx)

    def _keep_reliable_components_per_class(self, mask_2d: np.ndarray) -> np.ndarray:
        cleaned = np.zeros_like(mask_2d, dtype=np.int64)
        for class_id in range(1, NUM_CLASSES):
            class_mask = (mask_2d == class_id).astype(np.uint8)
            labeled, num_components = ndi.label(class_mask)
            if num_components == 0:
                continue
            component_ids = np.arange(1, num_components + 1)
            areas = np.array(ndi.sum(class_mask, labeled, index=component_ids), dtype=np.float64)
            largest_area = float(areas.max())
            keep_ids = [
                int(comp_id)
                for comp_id, area in zip(component_ids, areas)
                if area >= MIN_COMPONENT_PIXELS and area >= largest_area * SECONDARY_COMPONENT_RATIO
            ]
            if not keep_ids and largest_area >= MIN_COMPONENT_PIXELS:
                keep_ids = [int(component_ids[np.argmax(areas)])]
            for comp_id in keep_ids:
                cleaned[labeled == comp_id] = class_id
        return cleaned

    def _trim_edge_outliers(
        self,
        mask_2d: np.ndarray,
        stats: list[dict[str, float | int | str]],
        start_idx: int | None,
        end_idx: int | None,
    ) -> np.ndarray:
        if not stats or start_idx is None or end_idx is None:
            return mask_2d
        out = mask_2d.copy()
        median_area = float(np.median([float(row["area_pixels"]) for row in stats]))
        allowed_start = max(0, start_idx - ALLOW_MARGIN_LABELS)
        allowed_end = min(len(CANONICAL_LABELS) - 1, end_idx + ALLOW_MARGIN_LABELS)
        for row in stats:
            label_index = int(row["label_index"])
            if allowed_start <= label_index <= allowed_end or median_area <= 0:
                continue
            if float(row["area_pixels"]) < median_area * MIN_AREA_RATIO_TO_KEEP_EDGE_OUTLIER:
                out[out == int(row["class_id"])] = 0
        return out

    def _enforce_monotonic_centroids(self, mask_2d: np.ndarray) -> np.ndarray:
        out = mask_2d.copy()
        while True:
            stats = class_stats_from_mask(out)
            if len(stats) <= 1:
                break
            violation_found = False
            for prev_row, curr_row in zip(stats[:-1], stats[1:]):
                if float(curr_row["centroid_y"]) + MONOTONIC_TOLERANCE_PX < float(prev_row["centroid_y"]):
                    drop_class = (
                        int(prev_row["class_id"])
                        if float(prev_row["area_pixels"]) < float(curr_row["area_pixels"])
                        else int(curr_row["class_id"])
                    )
                    out[out == drop_class] = 0
                    violation_found = True
                    break
            if not violation_found:
                break
        return out

    def _restore_supported_edge_labels(
        self,
        mask_2d: np.ndarray,
        reference_mask: np.ndarray,
        start_idx: int | None,
        end_idx: int | None,
    ) -> np.ndarray:
        if start_idx is None or end_idx is None:
            return mask_2d
        out = mask_2d.copy()
        ref_stats = class_stats_from_mask(reference_mask)
        final_stats = class_stats_from_mask(out)
        if not ref_stats:
            return out
        median_area = float(np.median([float(row["area_pixels"]) for row in ref_stats]))
        kept_indices = {int(row["label_index"]) for row in final_stats}
        boundary_candidates = {max(0, start_idx - 1), min(len(CANONICAL_LABELS) - 1, end_idx + 1)}
        for row in ref_stats:
            label_index = int(row["label_index"])
            if label_index in kept_indices or label_index not in boundary_candidates:
                continue
            if abs(label_index - start_idx) > RESTORE_NEIGHBOR_DISTANCE and abs(label_index - end_idx) > RESTORE_NEIGHBOR_DISTANCE:
                continue
            if median_area <= 0 or float(row["area_pixels"]) < median_area * RESTORE_EDGE_MIN_AREA_RATIO:
                continue
            class_id = int(row["class_id"])
            out[reference_mask == class_id] = class_id
        return out

    def _estimate_last_visible_from_mask(self, pred_mask: np.ndarray) -> int:
        present_indices = [
            CANONICAL_LABELS.index(CLASS_NAMES[class_id])
            for class_id in sorted(int(x) for x in np.unique(pred_mask) if int(x) > 0)
            if CLASS_NAMES[class_id] in CANONICAL_LABELS
        ]
        if not present_indices:
            return 0
        return int(max(present_indices))

    def _blend_last_prediction(self, logits: torch.Tensor, heuristic_last_idx: int) -> int:
        probs = torch.softmax(logits, dim=1)[0]
        argmax_idx = int(torch.argmax(probs).item())
        class_axis = torch.arange(probs.shape[0], dtype=torch.float32, device=probs.device)
        expected_idx = float((probs * class_axis).sum().item())
        blended = (
            (1.0 - LAST_EXPECTATION_BLEND - LAST_HEURISTIC_BLEND) * argmax_idx
            + LAST_EXPECTATION_BLEND * expected_idx
            + LAST_HEURISTIC_BLEND * float(heuristic_last_idx)
        )
        return int(np.clip(round(blended), ASSUMED_FIRST_VISIBLE_IDX, len(CANONICAL_LABELS) - 1))

    def _clip_mask_to_last_idx(self, mask_2d: np.ndarray, first_idx: int, last_idx: int) -> np.ndarray:
        last_idx = int(max(last_idx, first_idx))
        allowed_labels = CANONICAL_LABELS[first_idx : last_idx + 1]
        allowed_ids = {LABEL_TO_CLASS_ID[label] for label in allowed_labels}
        out = np.zeros_like(mask_2d, dtype=np.int64)
        for class_id in allowed_ids:
            out[mask_2d == class_id] = class_id
        return out

    def _present_labels_from_mask(self, mask_2d: np.ndarray) -> list[str]:
        ids = sorted(int(x) for x in np.unique(mask_2d) if int(x) > 0)
        return [CLASS_NAMES[i] for i in ids]

    def _extract_vertebrae_geometry(self, mask_2d: np.ndarray) -> list[VertebraGeometry]:
        vertebrae: list[VertebraGeometry] = []
        for class_id in sorted(int(x) for x in np.unique(mask_2d) if int(x) > 0):
            y_coords, x_coords = np.where(mask_2d == class_id)
            if len(x_coords) == 0:
                continue

            x0 = int(x_coords.min())
            y0 = int(y_coords.min())
            x1 = int(x_coords.max()) + 1
            y1 = int(y_coords.max()) + 1
            centroid_x = float(x_coords.mean())
            centroid_y = float(y_coords.mean())
            orientation_degrees = self._orientation_from_points(x_coords, y_coords)

            vertebrae.append(
                VertebraGeometry(
                    label=CLASS_NAMES[class_id],
                    mask_id=class_id,
                    bbox=(x0, y0, x1, y1),
                    centroid=(round(centroid_x, 2), round(centroid_y, 2)),
                    area_pixels=int(len(x_coords)),
                    orientation_degrees=None if orientation_degrees is None else round(orientation_degrees, 2),
                )
            )

        return vertebrae

    def _orientation_from_points(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float | None:
        if len(x_coords) < 3:
            return None

        points = np.column_stack((x_coords.astype(np.float32), y_coords.astype(np.float32)))
        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        major_axis = vh[0]
        angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
        if angle > 90.0:
            angle -= 180.0
        if angle < -90.0:
            angle += 180.0
        return angle

    def _project_roi_mask_to_full_image(
        self,
        roi_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        full_mask = np.zeros(image_shape, dtype=np.uint8)
        resized_roi = resize_mask(roi_mask, (y1 - y0, x1 - x0)).astype(np.uint8)
        full_mask[y0:y1, x0:x1] = resized_roi
        return full_mask

    def _save_preview(
        self,
        image_raw: np.ndarray,
        full_mask: np.ndarray,
        preview_path: Path,
        labels: list[str],
        pred_last_label: str,
    ) -> None:
        image_rgb = np.stack([image_raw] * 3, axis=-1).astype(np.uint8)
        color_mask = COLOR_MAP[np.clip(full_mask, 0, len(COLOR_MAP) - 1)]
        alpha = (full_mask > 0)[..., None].astype(np.float32) * 0.45
        overlay = (image_rgb.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha).astype(np.uint8)

        preview = Image.fromarray(overlay)
        label_mask = Image.fromarray(full_mask.astype(np.uint8))
        max_height = 900
        if preview.height > max_height:
            ratio = max_height / preview.height
            size = (int(preview.width * ratio), max_height)
            preview = preview.resize(size, Image.BILINEAR)
            label_mask = label_mask.resize(size, Image.NEAREST)

        canvas = preview.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        try:
            label_font = ImageFont.load_default(size=18)
        except TypeError:
            label_font = ImageFont.load_default()
        mask_array = np.asarray(label_mask)
        for class_id in sorted(int(x) for x in np.unique(mask_array) if int(x) > 0):
            label = CLASS_NAMES[class_id]
            y_coords, x_coords = np.where(mask_array == class_id)
            if len(x_coords) == 0:
                continue

            x = int(np.median(x_coords))
            y = int(np.median(y_coords))
            text_bbox = draw.textbbox((0, 0), label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            box_x0 = max(6, min(x + 8, canvas.width - text_width - 14))
            box_y0 = max(6, min(y - text_height // 2 - 5, canvas.height - text_height - 10))
            box_x1 = box_x0 + text_width + 12
            box_y1 = box_y0 + text_height + 8

            draw.rounded_rectangle((box_x0, box_y0, box_x1, box_y1), radius=6, fill=(255, 255, 255), outline=(13, 132, 121), width=2)
            draw.text((box_x0 + 6, box_y0 + 4), label, fill=(18, 55, 92), font=label_font)

        preview_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(preview_path)
