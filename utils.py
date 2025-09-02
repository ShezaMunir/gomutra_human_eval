import os
import json
import re
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from datetime import datetime, date

# Bracketed tag like [Appeal to Authority]
TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")

def sanitize_username(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip())
    return safe or "annotator"

def load_dataset(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    # Ensure key columns exist
    default_cols = [
        "Transcript No", "URL", "Original Text", "English Translation",
        "gpt4o_layer2_annotations", "gpt4_layer2_annotations",
        "gpt41_layer2_annotations", "gpt5_layer2_annotations",
    ]
    for c in default_cols:
        if c not in df.columns:
            df[c] = None
    return df

def discover_models(df: pd.DataFrame) -> List[str]:
    candidates = [c for c in df.columns if c.endswith("_layer2_annotations")]
    order = [
        "gpt4o_layer2_annotations",
        "gpt4_layer2_annotations",
        "gpt41_layer2_annotations",
        "gpt5_layer2_annotations",
    ]
    ordered = [c for c in order if c in candidates]
    remaining = [c for c in candidates if c not in ordered]
    return ordered + sorted(remaining)

def ensure_user_dirs(base_dir: str, username: str, models: List[str]) -> str:
    user_dir = os.path.join(base_dir, sanitize_username(username))
    os.makedirs(user_dir, exist_ok=True)
    for m in models:
        os.makedirs(os.path.join(user_dir, m), exist_ok=True)
    return user_dir

def json_path_for(user_dir: str, model: str, row_index: int) -> str:
    return os.path.join(user_dir, model, f"T{row_index+1}.json")

# ---------- JSON helpers ----------
def _to_jsonable(obj):
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # pandas specials
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # datetime
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    # containers
    if isinstance(obj, dict):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # fallback
    return str(obj)

def load_existing_annotations(user_dir: str, model: str, row_index: int) -> Dict[str, Any]:
    path = json_path_for(user_dir, model, row_index)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        try:
            os.replace(path, path + ".corrupt")
        except Exception:
            pass
        return {}
    except Exception:
        return {}

def save_annotations(user_dir: str, model: str, row_index: int, payload: Dict[str, Any]) -> None:
    path = json_path_for(user_dir, model, row_index)
    safe = _to_jsonable(payload)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

# ---------- Inline tag extraction/rendering ----------
def extract_tags(tagged_text: str) -> List[Dict[str, Any]]:
    """Return list of tags with their order and text.
    Each item: {"idx": n (1..N), "text": tag_text, "start": pos, "end": pos}
    """
    if not isinstance(tagged_text, str) or not tagged_text.strip():
        return []
    tags = []
    for i, m in enumerate(TAG_PATTERN.finditer(tagged_text), start=1):
        tags.append({
            "idx": i,
            "text": m.group(1).strip(),
            "start": m.start(),
            "end": m.end(),
        })
    return tags

def build_inline_stream(tagged_text: str) -> List[Dict[str, Any]]:
    """Build a stream alternating between text chunks and tag chips to render inline.
    Returns a list of dicts with shape:
      {"kind": "text", "text": str} or {"kind": "tag", "text": tag_text, "idx": int}
    """
    if not isinstance(tagged_text, str):
        tagged_text = ""
    tags = extract_tags(tagged_text)
    if not tags:
        return [{"kind": "text", "text": tagged_text}]

    stream: List[Dict[str, Any]] = []
    last = 0
    for t in tags:
        if t["start"] > last:
            stream.append({"kind": "text", "text": tagged_text[last:t["start"]]})
        stream.append({"kind": "tag", "text": t["text"], "idx": t["idx"]})
        last = t["end"]
    if last < len(tagged_text):
        stream.append({"kind": "text", "text": tagged_text[last:]})
    return stream

def count_tags_in_text(tagged_text: str) -> int:
    if not isinstance(tagged_text, str):
        return 0
    return len(TAG_PATTERN.findall(tagged_text))

def build_payload_inline(
    username: str,
    model: str,
    row_index: int,
    transcript_no: Any,
    df_row: pd.Series,
    tags: List[Dict[str, Any]],
    form_data: Dict[str, Any],
    notes: str,
) -> Dict[str, Any]:
    items = []
    for t in tags:
        key = f"dec_tag_{t['idx']}"
        decision = form_data.get(key)  # "agree" | "disagree" | None
        items.append({
            "tag_index": int(t["idx"]),
            "tag_text": t["text"],
            "decision": decision,
        })
    return {
        "annotator": username,
        "model": model,
        "row_index": _safe_int(row_index, None),
        "transcript_no": _safe_int(transcript_no, None),
        "title": df_row.get("Title", None),
        "stance": df_row.get("Stance", None),
        "items": items,
        "notes": notes,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
