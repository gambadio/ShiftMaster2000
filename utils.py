from __future__ import annotations
import io, json, re
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
from models import Project
from prompt_templates import build_system_prompt

# ----------------------------
# Project I/O
# ----------------------------
def save_project(path: str, project: Project) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project.model_dump(), f, ensure_ascii=False, indent=2)

def load_project_dict(data: Dict[str, Any]) -> Project:
    return Project.model_validate(data)

# ----------------------------
# Prompt compiler
# ----------------------------
def compile_prompt(project: Project, schedule_payload: Optional[Dict[str, Any]] = None, today_iso: Optional[str] = None) -> str:
    return build_system_prompt(project, schedule_payload=schedule_payload, today_iso=today_iso)

# ----------------------------
# OpenAI-compatible call
# ----------------------------
import requests
def call_llm(
    prompt: str,
    endpoint: str,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    timeout: int = 60,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "Produce the schedule now."}],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------
# Schedule file parsing
# ----------------------------

DATE_COL_CANDIDATES = ["date", "datum", "day", "datum_tag"]
FROM_COL_CANDIDATES = ["date_from", "from", "von"]
TO_COL_CANDIDATES   = ["date_to", "to", "bis"]
EMP_COL_CANDIDATES  = ["employee", "employee_id", "name", "mitarbeiter", "person"]
ROLE_COL_CANDIDATES = ["role", "position", "funktion", "bereich"]
SHIFT_COL_CANDS     = ["shift", "schicht"]
START_COL_CANDS     = ["start", "start_time", "beginn"]
END_COL_CANDS       = ["end", "end_time", "ende"]
TYPE_COL_CANDS      = ["type", "event", "reason", "typ", "art"]
NOTES_COL_CANDS     = ["notes", "note", "bemerkung", "comment"]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: re.sub(r"[^a-z0-9]+","_", c.strip().lower()) for c in df.columns}
    return df.rename(columns=m)

def _pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _parse_date(val) -> Optional[date]:
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, )):
        return val.date()
    if isinstance(val, datetime):
        return val.date()
    s = str(val).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, dayfirst=False, utc=False).date()
    except Exception:
        try:
            return pd.to_datetime(s, dayfirst=True, utc=False).date()
        except Exception:
            return None

def _read_schedule_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(bio)
    else:
        # let pandas sniff separators; fallback to comma
        try:
            df = pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, sep=";")
    return _normalize_cols(df)

def _expand_ranges(df: pd.DataFrame, col_from: str, col_to: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        d1 = _parse_date(r.get(col_from))
        d2 = _parse_date(r.get(col_to))
        if not d1 and not d2:
            # nothing to expand, skip
            continue
        if d1 and not d2:
            d2 = d1
        if d2 and not d1:
            d1 = d2
        if d1 and d2 and d2 < d1:
            d1, d2 = d2, d1
        # n.b. guard long ranges to 366 days
        if d1 and d2:
            rng = pd.date_range(d1, d2, freq="D")[:366]
            for d in rng:
                rr = dict(r)
                rr["date"] = d.date().isoformat()
                rows.append(rr)
    if rows:
        out = pd.DataFrame(rows)
        return _normalize_cols(out)
    return df

def parse_schedule_to_payload(file_bytes: bytes, filename: str, today: date) -> Dict[str, Any]:
    """
    Returns a compact JSON payload with two top-level keys:
      - past_entries[]    (date < today)
      - future_entries[]  (date >= today)
    Each entry carries selected normalized fields.
    """
    df = _read_schedule_file(file_bytes, filename)
    # try to create a single date column
    col_date = _pick(df, DATE_COL_CANDIDATES)
    col_from = _pick(df, FROM_COL_CANDIDATES)
    col_to   = _pick(df, TO_COL_CANDIDATES)

    if not col_date and (col_from or col_to):
        df = _expand_ranges(df, col_from or col_to, col_to or col_from)
        col_date = "date"

    if not col_date and "date" not in df.columns:
        # last resort: look for something that parses as date in first non-null column
        for c in df.columns:
            maybe = _parse_date(df[c].dropna().iloc[0]) if not df[c].dropna().empty else None
            if maybe:
                col_date = c
                break

    if not col_date:
        raise ValueError("Could not detect a date/date_from/date_to column in the uploaded schedule file.")

    # pick optional fields
    col_emp  = _pick(df, EMP_COL_CANDIDATES)
    col_role = _pick(df, ROLE_COL_CANDIDATES)
    col_shift= _pick(df, SHIFT_COL_CANDS)
    col_start= _pick(df, START_COL_CANDS)
    col_end  = _pick(df, END_COL_CANDS)
    col_type = _pick(df, TYPE_COL_CANDS)
    col_notes= _pick(df, NOTES_COL_CANDS)

    # normalize row dicts
    def make_row(sr: pd.Series) -> Dict[str, Any]:
        d = _parse_date(sr[col_date])
        return {
            "date": d.isoformat() if d else None,
            **({"employee": str(sr[col_emp]).strip()} if col_emp else {}),
            **({"role": str(sr[col_role]).strip()} if col_role else {}),
            **({"shift": str(sr[col_shift]).strip()} if col_shift else {}),
            **({"start": str(sr[col_start]).strip()} if col_start else {}),
            **({"end": str(sr[col_end]).strip()} if col_end else {}),
            **({"type": str(sr[col_type]).strip()} if col_type else {}),
            **({"notes": str(sr[col_notes]).strip()} if col_notes else {}),
        }

    rows: List[Dict[str, Any]] = [make_row(r) for _, r in df.iterrows() if _parse_date(r[col_date]) is not None]

    # split by today
    past: List[Dict[str, Any]] = []
    future: List[Dict[str, Any]] = []
    for r in rows:
        d = pd.to_datetime(r["date"]).date()
        if d < today:
            past.append(r)
        else:
            future.append(r)

    # compact / cap to avoid massive prompts
    past_sorted = sorted(past, key=lambda x: x["date"], reverse=True)
    future_sorted = sorted(future, key=lambda x: x["date"])

    # Limit payload sizes sensibly (adjust as needed)
    past_cap = past_sorted[:800]
    future_cap = future_sorted[:800]

    # derive lightweight fairness hints (last 14 days)
    last_14 = [r for r in past_sorted if (today - pd.to_datetime(r["date"]).date()).days <= 14]
    fairness_hints = _compute_fairness_hints(last_14)

    return {
        "meta": {
            "source_filename": filename,
            "today": today.isoformat(),
            "rows_in_file": len(rows),
            "rows_past_included": len(past_cap),
            "rows_future_included": len(future_cap),
        },
        "past_entries": past_cap,
        "future_entries": future_cap,
        "fairness_hints": fairness_hints,  # e.g., recent late/pikett streaks per employee
    }

def _compute_fairness_hints(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Very lightweight heuristic summaries for rotation fairness:
    counts of entries labeled as 'late' or 'pikett' in the recent window.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        emp = r.get("employee") or "unknown"
        role = (r.get("role") or "").lower()
        shift = (r.get("shift") or "").lower()
        label_late = "late" in shift or "spät" in shift or ("10:00" in (r.get("start") or "") and "19:00" in (r.get("end") or ""))
        label_pikett = "pikett" in role or "pikett" in shift or "piket" in role or "piket" in shift
        if emp not in counts:
            counts[emp] = {"late_14d": 0, "pikett_14d": 0}
        if label_late:
            counts[emp]["late_14d"] += 1
        if label_pikett:
            counts[emp]["pikett_14d"] += 1
    return counts
