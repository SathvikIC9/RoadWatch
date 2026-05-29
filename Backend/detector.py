# detector.py
# ============================================================
# RoadWatch CV Engine — loads trained YOLOv8 and runs inference
# Works on CPU (your i5 laptop, no GPU needed)
# ============================================================

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import time

# Try ultralytics first (easier), fall back to ONNX runtime
try:
    from ultralytics import YOLO
    USE_ULTRALYTICS = True
except ImportError:
    import onnxruntime as ort
    USE_ULTRALYTICS = False


# ── Class definitions ─────────────────────────────────────────
CLASS_NAMES = {
    0: 'longitudinal_crack',
    1: 'transverse_crack',
    2: 'alligator_crack',
    3: 'pothole'
}

SEVERITY_MAP = {
    'pothole':            {'level': 'HIGH',   'score': 3, 'priority': 1},
    'alligator_crack':    {'level': 'HIGH',   'score': 3, 'priority': 2},
    'transverse_crack':   {'level': 'MEDIUM', 'score': 2, 'priority': 3},
    'longitudinal_crack': {'level': 'LOW',    'score': 1, 'priority': 4},
}

DESCRIPTIONS = {
    'pothole':            'Pothole detected — road surface has a hollow depression',
    'alligator_crack':    'Alligator cracking — severe mesh of cracks, road near failure',
    'transverse_crack':   'Transverse crack — crack running across the road width',
    'longitudinal_crack': 'Longitudinal crack — crack running along the road length',
}

# Per-class confidence overrides — potholes are under-represented in RDD2022
# so we accept lower confidence for them specifically.
CLASS_CONF_OVERRIDE = {
    'pothole': 0.02,         # more sensitive for potholes
    'alligator_crack': 0.03, # alligator cracks also benefit from lower floor
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list          # [x1, y1, x2, y2] in pixels
    severity_level: str # HIGH / MEDIUM / LOW
    severity_score: int # 1, 2, or 3
    description: str


@dataclass
class InferenceResult:
    detections: list[Detection]
    overall_severity: str
    overall_score: int
    defect_count: int
    inference_time_ms: float
    image_width: int
    image_height: int
    annotated_image: Optional[np.ndarray]


class RoadDamageDetector:
    """
    Loads trained YOLOv8 weights and runs road damage inference.
    Put your .pt or .onnx file in the models/ directory.
    """

    def __init__(
        self,
        model_path: str = "models/FINAL_best.pt",
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Place FINAL_best.pt (or .onnx) in the models/ folder."
            )

        print(f"Loading model from {model_path}...")
        self._load_model()
        print(f"✅ Model loaded. Backend: {'Ultralytics' if USE_ULTRALYTICS else 'ONNX Runtime'}")

    # ── Model loading ─────────────────────────────────────────

    def _load_model(self):
        if USE_ULTRALYTICS and self.model_path.suffix == '.pt':
            self.model = YOLO(str(self.model_path))
            self.backend = 'ultralytics'
        else:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.backend = 'onnx'

    # ── ONNX preprocessing / postprocessing ──────────────────

    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # add batch dim
        return img

    def _postprocess_onnx(
        self, output: np.ndarray, orig_w: int, orig_h: int
    ) -> list[dict]:
        """
        Parse raw YOLOv8 ONNX output.
        Output shape: [1, 8400, 4+nc] after transpose → iterate columns.
        Coords are in resized pixel space (0–input_size), NOT 0-1 normalised.
        """
        predictions = output[0][0]  # [4+nc, 8400]
        detections = []

        for pred in predictions.T:          # iterate over 8400 anchor proposals
            x, y, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            cls_name = CLASS_NAMES.get(class_id, 'unknown')

            # Use per-class threshold if defined, else global threshold
            threshold = CLASS_CONF_OVERRIDE.get(cls_name, self.conf_threshold)
            if confidence < threshold:
                continue

            # Scale from resized (input_size) space back to original image size
            x1 = int((x - w / 2) * orig_w / self.input_size)
            y1 = int((y - h / 2) * orig_h / self.input_size)
            x2 = int((x + w / 2) * orig_w / self.input_size)
            y2 = int((y + h / 2) * orig_h / self.input_size)

            detections.append({
                'class_id': class_id,
                'class_name': cls_name,
                'confidence': confidence,
                'bbox': [max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)]
            })

        return detections

    def _nms(self, detections: list[dict], iou_threshold: float) -> list[dict]:
        """
        Class-aware greedy NMS.
        Boxes from DIFFERENT classes are never suppressed against each other —
        this is critical when a pothole sits inside an alligator crack region.
        """
        if not detections:
            return []

        # Group by class and run NMS within each class independently
        from collections import defaultdict
        by_class = defaultdict(list)
        for det in detections:
            by_class[det['class_id']].append(det)

        kept = []
        for class_dets in by_class.values():
            class_dets = sorted(class_dets, key=lambda d: d['confidence'], reverse=True)
            class_kept = []
            for det in class_dets:
                x1, y1, x2, y2 = det['bbox']
                suppressed = False
                for k in class_kept:
                    kx1, ky1, kx2, ky2 = k['bbox']
                    ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                    ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
                    if union > 0 and inter / union > iou_threshold:
                        suppressed = True
                        break
                if not suppressed:
                    class_kept.append(det)
            kept.extend(class_kept)

        return kept

    # ── Annotation drawing ────────────────────────────────────

    def _draw_annotations(
        self, image: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        img = image.copy()
        color_map = {
            'HIGH':   (0, 0, 255),    # red
            'MEDIUM': (0, 165, 255),  # orange
            'LOW':    (0, 255, 0),    # green
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.severity_level, (200, 200, 200))
            label = f"{det.class_name.replace('_', ' ')} {det.confidence:.0%}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    # ── Main inference ────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        annotate: bool = True
    ) -> InferenceResult:
        """
        Run inference on a single BGR numpy image.
        Returns InferenceResult with all detections and severity.
        """
        orig_h, orig_w = image.shape[:2]
        t_start = time.time()
        raw_detections = []

        if self.backend == 'ultralytics':
            # Pass per-class conf as the global floor; ultralytics handles its own NMS
            results = self.model.predict(
                source=image,
                conf=min(CLASS_CONF_OVERRIDE.values()),  # use lowest override as floor
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False,
                agnostic_nms=False,  # class-aware NMS — keeps pothole + alligator_crack
            )
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                    # Post-filter with per-class threshold
                    threshold = CLASS_CONF_OVERRIDE.get(cls_name, self.conf_threshold)
                    if conf < threshold:
                        continue
                    raw_detections.append({
                        'class_id':   cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': [int(v) for v in box.xyxy[0].tolist()]
                    })

        else:  # ONNX
            inp = self._preprocess_onnx(image)
            output = self.session.run(None, {self.input_name: inp})
            raw_detections = self._postprocess_onnx(output, orig_w, orig_h)
            raw_detections = self._nms(raw_detections, self.iou_threshold)

        inference_ms = (time.time() - t_start) * 1000

        # Build Detection objects
        detections = []
        for rd in raw_detections:
            cls_name = rd.get('class_name') or CLASS_NAMES.get(rd['class_id'], 'unknown')
            sev = SEVERITY_MAP.get(cls_name, {'level': 'LOW', 'score': 1})
            detections.append(Detection(
                class_name=cls_name,
                confidence=rd['confidence'],
                bbox=rd['bbox'],
                severity_level=sev['level'],
                severity_score=sev['score'],
                description=DESCRIPTIONS.get(cls_name, ''),
            ))

        detections.sort(key=lambda d: (-d.severity_score, -d.confidence))

        overall_score = max((d.severity_score for d in detections), default=0)
        SCORE_TO_LEVEL = {3: 'HIGH', 2: 'MEDIUM', 1: 'LOW', 0: 'NONE'}
        overall_severity = SCORE_TO_LEVEL.get(overall_score, 'NONE')

        annotated = self._draw_annotations(image, detections) if annotate else None

        return InferenceResult(
            detections=detections,
            overall_severity=overall_severity,
            overall_score=overall_score,
            defect_count=len(detections),
            inference_time_ms=inference_ms,
            image_width=orig_w,
            image_height=orig_h,
            annotated_image=annotated,
        )

    def predict_from_path(self, image_path: str, annotate: bool = True) -> InferenceResult:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.predict(img, annotate)

    def predict_from_bytes(self, image_bytes: bytes, annotate: bool = True) -> InferenceResult:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return self.predict(img, annotate)