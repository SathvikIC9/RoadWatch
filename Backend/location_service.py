# location_service.py
# ============================================================
# RoadWatch Location Service
# GPS coordinates → road name → road type → authority routing
# Uses Google Maps Geocoding API + Roads API
# ============================================================

import re
import requests
from dataclasses import dataclass
from typing import Optional


NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {
    "User-Agent": "RoadWatchApp/1.0 (contact@roadwatch.in)",
    "Accept-Language": "en",
}

# ── Road type detection patterns ──────────────────────────────
# These regex patterns identify road type from Google Maps road name

NH_PATTERNS  = [r'\bNH[-\s]?\d+', r'National Highway', r'NH\d+']
SH_PATTERNS  = [r'\bSH[-\s]?\d+', r'State Highway', r'SH\d+']
MDR_PATTERNS = [r'\bMDR[-\s]?\d+', r'Major District Road']
CITY_KEYWORDS = [
    'road', 'street', 'avenue', 'lane', 'colony', 'nagar',
    'marg', 'path', 'circle', 'cross', 'layout', 'enclave'
]


@dataclass
class LocationInfo:
    latitude: float
    longitude: float
    formatted_address: str
    road_name: str
    road_type: str           # NH / SH / MDR / Rural / City / Unknown
    state: str
    district: str
    city: str
    pincode: str
    authority: str           # Who manages this road
    authority_helpline: str
    authority_portal: str
    confidence: str          # HIGH / MEDIUM / LOW


def detect_road_type(road_name: str, full_address: str) -> tuple[str, str]:
    """
    Detect road type and responsible authority from road name.
    Returns (road_type, authority_name)
    """
    combined = f"{road_name} {full_address}".upper()

    # Check NH first — most specific
    for pattern in NH_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return "NH", "NHAI"

    # Check SH
    for pattern in SH_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return "SH", "State PWD"

    # Check MDR
    for pattern in MDR_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return "MDR", "District Collector"

    # Check if inside city (urban keywords)
    road_lower = road_name.lower()
    if any(kw in road_lower for kw in CITY_KEYWORDS):
        return "City", "Municipal Corporation"

    # Default: could be rural or unknown
    return "Unknown", "Local Authority"


def get_state_authority(road_type: str, state: str, city: str) -> tuple[str, str, str]:
    """
    Returns (authority_name, helpline, portal) based on road type + location.
    """
    state_lower = state.lower()
    city_lower  = city.lower()

    if road_type == "NH":
        return "NHAI", "1033", "https://rajmargyatra.nhai.org"

    if road_type == "City":
        # Map major cities to their corporations
        city_map = {
            "hyderabad":  ("GHMC",  "040-21111111",  "https://ghmc.gov.in/complaints"),
            "bengaluru":  ("BBMP",  "080-22660000",  "https://bbmpsahaaya.karnataka.gov.in"),
            "bangalore":  ("BBMP",  "080-22660000",  "https://bbmpsahaaya.karnataka.gov.in"),
            "mumbai":     ("BMC",   "1916",           "https://portal.mcgm.gov.in"),
            "chennai":    ("GCC",   "044-25384530",  "https://chennaicorporation.gov.in"),
            "delhi":      ("MCD",   "011-23227044",  "https://mcd.gov.in"),
            "kolkata":    ("KMC",   "033-22861000",  "https://kmcgov.in"),
            "pune":       ("PMC",   "020-25507388",  "https://pmc.gov.in"),
            "ahmedabad":  ("AMC",   "079-25391811",  "https://ahmedabadcity.gov.in"),
            "secunderabad": ("GHMC","040-21111111",  "https://ghmc.gov.in/complaints"),
        }
        for city_key, info in city_map.items():
            if city_key in city_lower:
                return info
        return "Municipal Corporation", "pgportal.gov.in", "https://pgportal.gov.in"

    if road_type in ("SH", "MDR"):
        state_map = {
            "telangana":    ("Telangana R&B Dept",    "040-24651100",  "https://rb.telangana.gov.in"),
            "andhra pradesh":("AP Roads & Buildings", "0863-2340000",  "https://rd.ap.gov.in"),
            "karnataka":    ("Karnataka PWD",         "080-22032070",  "https://pwd.karnataka.gov.in"),
            "maharashtra":  ("Maharashtra PWD",       "022-22024713",  "https://mahaonline.gov.in"),
            "tamil nadu":   ("TN Highways Dept",      "044-28270101",  "https://highways.tn.gov.in"),
            "kerala":       ("Kerala PWD",            "0471-2518001",  "https://pwd.kerala.gov.in"),
            "uttar pradesh":("UP PWD",                "0522-2230882",  "https://up.gov.in"),
            "rajasthan":    ("Rajasthan PWD",         "0141-2227256",  "https://pwd.rajasthan.gov.in"),
            "gujarat":      ("Gujarat R&B Dept",      "079-23250871",  "https://roads.gujarat.gov.in"),
            "madhya pradesh":("MP PWD",               "0755-2441600",  "https://mppwd.mp.gov.in"),
            "west bengal":  ("WB PWD",                "033-22483215",  "https://wbpwd.gov.in"),
        }
        for state_key, info in state_map.items():
            if state_key in state_lower:
                return info
        return "State PWD", "pgportal.gov.in", "https://pgportal.gov.in"

    if road_type == "Rural":
        return "NRIDA (Meri Sadak App)", "helpdesk@pmgsy.nic.in", "https://omms.nic.in"

    return "Local Authority", "pgportal.gov.in", "https://pgportal.gov.in"


def get_location_info(lat: float, lng: float) -> LocationInfo:
    """
    Main function: takes GPS coordinates → returns full LocationInfo.
    Calls Nominatim (OpenStreetMap) reverse geocoding — free, no API key needed.
    Note: Nominatim has a 1 req/sec rate limit; add time.sleep(1) in batch usage.
    """
    params = {
        "lat": lat,
        "lon": lng,
        "format": "json",
        "addressdetails": 1,
        "zoom": 18,          # street-level detail
    }

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Nominatim API call failed: {e}")

    if "error" in data:
        raise ValueError(f"Nominatim error: {data['error']}")

    address   = data.get("address", {})
    formatted = data.get("display_name", "")

    # ── Parse address fields ──────────────────────────────────
    road_name = (
        address.get("road") or
        address.get("pedestrian") or
        address.get("footway") or
        address.get("path") or
        ""
    )
    city     = address.get("city") or address.get("town") or address.get("village") or ""
    district = address.get("county") or address.get("state_district") or ""
    state    = address.get("state") or ""
    pincode  = address.get("postcode") or ""

    # ── Detect road type ──────────────────────────────────────
    road_type, base_authority = detect_road_type(road_name, formatted)

    # ── Get specific authority based on location ──────────────
    authority, helpline, portal = get_state_authority(road_type, state, city)

    # ── Confidence ────────────────────────────────────────────
    if road_name and state and city:
        confidence = "HIGH"
    elif road_name or (state and city):
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return LocationInfo(
        latitude=lat,
        longitude=lng,
        formatted_address=formatted,
        road_name=road_name or "Unknown road",
        road_type=road_type,
        state=state,
        district=district,
        city=city,
        pincode=pincode,
        authority=authority,
        authority_helpline=helpline,
        authority_portal=portal,
        confidence=confidence,
    )


def get_nearby_roads(lat: float, lng: float, radius_m: int = 100) -> list[dict]:
    """
    Get nearby road names using Nominatim search.
    Returns a list of road dicts near the given coordinates.
    Note: Nominatim rate limit is 1 req/sec — add sleep in loops.
    """
    params = {
        "q": "road",
        "lat": lat,
        "lon": lng,
        "format": "json",
        "addressdetails": 1,
        "limit": 5,
        "featureType": "road",
    }
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        data = resp.json()
        return [
            {
                "road_name": item.get("address", {}).get("road", item.get("display_name", "")),
                "location": {"lat": float(item["lat"]), "lng": float(item["lon"])},
            }
            for item in data
            if item.get("lat")
        ]
    except Exception:
        return []