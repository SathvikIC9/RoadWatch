# main.py — RoadWatch Flask Backend (Complete)
# Endpoints: /health /detect /ask /location /road /contractor /authority /rag/rebuild
# Run: python main.py

import os, base64, uuid
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from detector import RoadDamageDetector
from rag_engine import RoadWatchRAG
from location_service import get_location_info

app = Flask(__name__)
CORS(app)

# ── Load models ───────────────────────────────────────────────
MODEL_PATH = "models/FINAL_best.pt"

try:
    detector = RoadDamageDetector(model_path=MODEL_PATH, conf_threshold=0.05, iou_threshold=0.45)
    print("✅ CV Detector ready")
except Exception as e:
    detector = None
    print(f"⚠️  CV Model: {e}")

try:
    rag = RoadWatchRAG()
except Exception as e:
    rag = None
    print(f"⚠️  RAG: {e}")


def build_summary(result):
    if result.defect_count == 0:
        return "No road damage detected in the uploaded image."
    types = list({d.class_name.replace('_', ' ') for d in result.detections})
    return f"Road damage detected: {', '.join(types)}. Severity: {result.overall_severity}. {result.defect_count} defect(s) found."


# ════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "cv_model": detector is not None, "rag": rag is not None})


# ════════════════════════════════════════════════════════════
# CV DETECTION
# ════════════════════════════════════════════════════════════
@app.route('/detect', methods=['POST'])
def detect():
    if detector is None:
        return jsonify({"error": "CV model not loaded"}), 503
    if 'file' not in request.files:
        return jsonify({"error": "No file sent"}), 400

    file = request.files['file']
    if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        return jsonify({"error": "Send JPG or PNG only"}), 400

    image_bytes = file.read()
    road_type   = request.form.get('road_type')
    return_ann  = request.form.get('return_annotated', 'true').lower() == 'true'

    try:
        result = detector.predict_from_bytes(image_bytes, annotate=return_ann)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ann_b64 = None
    if return_ann and result.annotated_image is not None:
        _, buf = cv2.imencode('.jpg', result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

    priority_map = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'NONE': 4}
    resp = {
        "success": True,
        "defect_count": result.defect_count,
        "overall_severity": result.overall_severity,
        "overall_score": result.overall_score,
        "inference_time_ms": round(result.inference_time_ms, 2),
        "complaint_summary": build_summary(result),
        "routing_priority": priority_map.get(result.overall_severity, 4),
        "annotated_image_b64": ann_b64,
        "detections": [
            {"class_name": d.class_name, "confidence": round(d.confidence, 4),
             "bbox": d.bbox, "severity_level": d.severity_level,
             "severity_score": d.severity_score, "description": d.description}
            for d in result.detections
        ],
    }

    # Auto-fetch authority info if RAG loaded and damage found
    if rag and result.defect_count > 0 and road_type:
        resp["authority"] = rag.get_authority_for_location(road_type)

    return jsonify(resp)


# ════════════════════════════════════════════════════════════
# LOCATION — GPS to road info
# ════════════════════════════════════════════════════════════
@app.route('/location', methods=['POST'])
def location():
    """
    POST JSON: { "lat": 17.385, "lng": 78.4867, "session_id": "abc" }
    Returns road type, authority, contact details from GPS coordinates.
    """
    data = request.get_json()
    if not data or 'lat' not in data or 'lng' not in data:
        return jsonify({"error": "Send JSON with lat and lng fields"}), 400

    lat = float(data['lat'])
    lng = float(data['lng'])
    session_id = data.get('session_id', str(uuid.uuid4()))

    try:
        info = get_location_info(lat, lng)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 503

    # Get authority from RAG for richer response
    authority_detail = {}
    if rag:
        authority_detail = rag.get_authority_for_location(
            info.road_type, info.state, info.city
        )

    return jsonify({
        "success":           True,
        "session_id":        session_id,
        "latitude":          lat,
        "longitude":         lng,
        "formatted_address": info.formatted_address,
        "road_name":         info.road_name,
        "road_type":         info.road_type,
        "state":             info.state,
        "district":          info.district,
        "city":              info.city,
        "pincode":           info.pincode,
        "confidence":        info.confidence,
        "authority": {
            "name":        authority_detail.get("body_name", info.authority),
            "short_name":  authority_detail.get("short_name", ""),
            "helpline":    authority_detail.get("helpline",   info.authority_helpline),
            "portal":      authority_detail.get("portal",     info.authority_portal),
            "email":       authority_detail.get("email",      ""),
            "sla_days":    authority_detail.get("sla_days",   45),
        }
    })


# ════════════════════════════════════════════════════════════
# RAG — Ask anything
# ════════════════════════════════════════════════════════════
@app.route('/ask', methods=['POST'])
def ask():
    """
    POST JSON:
    {
      "query": "Who do I contact for NH-44 pothole?",
      "session_id": "user-abc",   (optional — for conversation memory)
      "road_type": "NH",          (optional — from GPS)
      "road_name": "NH-44",       (optional)
      "state": "Telangana",       (optional)
      "city": "Hyderabad"         (optional)
    }
    """
    if rag is None:
        return jsonify({"error": "RAG not loaded"}), 503

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Send JSON with query field"}), 400

    query      = data.get('query', '').strip()
    session_id = data.get('session_id', 'default')
    road_type  = data.get('road_type')
    road_name  = data.get('road_name')
    state      = data.get('state')
    city       = data.get('city')

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    if len(query) > 500:
        return jsonify({"error": "Query too long. Max 500 chars."}), 400

    try:
        result = rag.answer(query, session_id, road_type, road_name, state, city)
        return jsonify({"success": True, "query": query,
                        "session_id": session_id, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/session/clear', methods=['POST'])
def clear_session():
    """POST JSON: { "session_id": "abc" } — clears conversation history."""
    data = request.get_json()
    session_id = data.get('session_id', 'default') if data else 'default'
    if rag:
        rag.clear_session(session_id)
    return jsonify({"success": True, "message": f"Session {session_id} cleared"})


# ════════════════════════════════════════════════════════════
# QUICK LOOKUP ROUTES
# ════════════════════════════════════════════════════════════
@app.route('/road/<road_id>', methods=['GET'])
def get_road(road_id):
    if rag is None: return jsonify({"error": "RAG not loaded"}), 503
    result = rag.answer(f"Tell me about road {road_id} budget contractor condition complaints")
    raw = next((d for d in rag.docs if d.get("road_id") == road_id.upper()), None)
    return jsonify({"success": True, "road_id": road_id, "summary": result["answer"], "raw": raw})


@app.route('/contractor/<contractor_id>', methods=['GET'])
def get_contractor(contractor_id):
    if rag is None: return jsonify({"error": "RAG not loaded"}), 503
    result = rag.answer(f"Tell me about contractor {contractor_id} quality score warranty breaches performance")
    raw = next((d for d in rag.docs if d.get("contractor_id") == contractor_id.upper()), None)
    return jsonify({"success": True, "contractor_id": contractor_id, "summary": result["answer"], "raw": raw})


@app.route('/authority', methods=['GET'])
def get_authority():
    """GET /authority?road_type=NH&state=Telangana&city=Hyderabad"""
    if rag is None: return jsonify({"error": "RAG not loaded"}), 503
    road_type = request.args.get('road_type', '')
    state     = request.args.get('state', '')
    city      = request.args.get('city', '')
    if not road_type:
        return jsonify({"error": "Send ?road_type=NH (or SH, MDR, City, Rural)"}), 400
    result = rag.get_authority_for_location(road_type, state, city)
    return jsonify({"success": True, "road_type": road_type, **result})


@app.route('/rag/rebuild', methods=['POST'])
def rebuild():
    if rag is None: return jsonify({"error": "RAG not loaded"}), 503
    try:
        rag.rebuild_index()
        return jsonify({"success": True, "message": "Index rebuilt"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
