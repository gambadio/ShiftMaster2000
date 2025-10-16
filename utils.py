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

DATE_COL_CANDIDATES = ["date", "datum", "day", "datum_tag", "startdatum"]
FROM_COL_CANDIDATES = ["date_from", "from", "von", "startdatum"]
TO_COL_CANDIDATES   = ["date_to", "to", "bis", "enddatum"]
EMP_COL_CANDIDATES  = ["employee", "employee_id", "name", "mitarbeiter", "person", "mitglied"]
EMAIL_COL_CANDIDATES = ["email", "e_mail", "e_mail_geschaftlich", "business_email"]
GROUP_COL_CANDIDATES = ["group", "gruppe", "team", "abteilung"]
ROLE_COL_CANDIDATES = ["role", "position", "funktion", "bereich", "bezeichnung"]
SHIFT_COL_CANDS     = ["shift", "schicht"]
START_TIME_CANDS    = ["start", "start_time", "beginn", "startzeit"]
END_TIME_CANDS      = ["end", "end_time", "ende", "endzeit"]
TYPE_COL_CANDS      = ["type", "event", "reason", "typ", "art", "grund_fur_arbeitsfreie_zeit"]
NOTES_COL_CANDS     = ["notes", "note", "bemerkung", "comment", "notizen"]
COLOR_COL_CANDS     = ["color", "colour", "themenfarbe", "farbe"]
BREAK_COL_CANDS     = ["break", "pause", "unpaid_break", "unbezahlte_pause_minuten"]
SHARED_COL_CANDS    = ["shared", "geteilt"]

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

# ----------------------------
# Dual-file Teams import
# ----------------------------
def parse_dual_schedule_files(
    shifts_file_bytes: bytes,
    shifts_filename: str,
    timeoff_file_bytes: Optional[bytes],
    timeoff_filename: Optional[str],
    today: date
) -> Dict[str, Any]:
    """
    Parse Microsoft Teams Shifts and Time-Off exports.

    Args:
        shifts_file_bytes: Shifts Excel file
        shifts_filename: Name of shifts file
        timeoff_file_bytes: Optional time-off Excel file
        timeoff_filename: Optional name of time-off file
        today: Reference date for splitting past/future

    Returns:
        Unified payload with past/future entries categorized by type
    """
    all_entries: List[Dict[str, Any]] = []

    # Parse shifts file
    df_shifts = _read_schedule_file(shifts_file_bytes, shifts_filename)
    shifts_entries = _parse_teams_file(df_shifts, entry_type="shift")
    all_entries.extend(shifts_entries)

    # Parse time-off file if provided
    if timeoff_file_bytes and timeoff_filename:
        df_timeoff = _read_schedule_file(timeoff_file_bytes, timeoff_filename)
        timeoff_entries = _parse_teams_file(df_timeoff, entry_type="time_off")
        all_entries.extend(timeoff_entries)

    # Split by today
    past: List[Dict[str, Any]] = []
    future: List[Dict[str, Any]] = []

    for entry in all_entries:
        entry_date = pd.to_datetime(entry["start_date"]).date()
        if entry_date < today:
            past.append(entry)
        else:
            future.append(entry)

    # Sort and cap
    past_sorted = sorted(past, key=lambda x: x["start_date"], reverse=True)
    future_sorted = sorted(future, key=lambda x: x["start_date"])

    past_cap = past_sorted[:800]
    future_cap = future_sorted[:800]

    # Compute fairness hints from last 14 days
    last_14 = [r for r in past_sorted if (today - pd.to_datetime(r["start_date"]).date()).days <= 14]
    fairness_hints = _compute_fairness_hints(last_14)

    return {
        "meta": {
            "source_shifts_file": shifts_filename,
            "source_timeoff_file": timeoff_filename if timeoff_filename else "Not provided",
            "today": today.isoformat(),
            "total_entries": len(all_entries),
            "rows_past_included": len(past_cap),
            "rows_future_included": len(future_cap),
        },
        "past_entries": past_cap,
        "future_entries": future_cap,
        "fairness_hints": fairness_hints,
    }

def _parse_teams_file(df: pd.DataFrame, entry_type: str = "shift") -> List[Dict[str, Any]]:
    """Parse a Teams export file (shifts or time-off) into entries"""
    # Pick columns
    col_emp = _pick(df, EMP_COL_CANDIDATES)
    col_email = _pick(df, EMAIL_COL_CANDIDATES)
    col_group = _pick(df, GROUP_COL_CANDIDATES)
    col_start_date = _pick(df, FROM_COL_CANDIDATES) or _pick(df, DATE_COL_CANDIDATES)
    col_start_time = _pick(df, START_TIME_CANDS)
    col_end_date = _pick(df, TO_COL_CANDIDATES)
    col_end_time = _pick(df, END_TIME_CANDS)
    col_color = _pick(df, COLOR_COL_CANDS)
    col_label = _pick(df, ROLE_COL_CANDIDATES)
    col_break = _pick(df, BREAK_COL_CANDS)
    col_notes = _pick(df, NOTES_COL_CANDS)
    col_shared = _pick(df, SHARED_COL_CANDS)
    col_reason = _pick(df, TYPE_COL_CANDS)

    if not col_emp or not col_start_date:
        raise ValueError(f"Could not detect employee or date columns in {entry_type} file")

    entries = []
    for _, row in df.iterrows():
        start_date = _parse_date(row[col_start_date])
        end_date = _parse_date(row[col_end_date]) if col_end_date else start_date

        if not start_date:
            continue

        # Extract employee name (remove quotes if present)
        emp_name = str(row[col_emp]).strip().strip('"')

        # Parse color code from Teams format (e.g., "1. Weiß" -> "1")
        color_code = None
        if col_color and not pd.isna(row[col_color]):
            color_str = str(row[col_color]).strip()
            if color_str and color_str[0].isdigit():
                color_code = color_str.split('.')[0]

        entry = {
            "employee": emp_name,
            "email": str(row[col_email]).strip() if col_email and not pd.isna(row[col_email]) else None,
            "group": str(row[col_group]).strip() if col_group and not pd.isna(row[col_group]) else None,
            "start_date": start_date.isoformat(),
            "start_time": str(row[col_start_time]).strip() if col_start_time and not pd.isna(row[col_start_time]) else None,
            "end_date": end_date.isoformat() if end_date else start_date.isoformat(),
            "end_time": str(row[col_end_time]).strip() if col_end_time and not pd.isna(row[col_end_time]) else None,
            "color_code": color_code,
            "label": str(row[col_label]).strip() if col_label and not pd.isna(row[col_label]) else None,
            "unpaid_break": int(row[col_break]) if col_break and not pd.isna(row[col_break]) else None,
            "notes": str(row[col_notes]).strip() if col_notes and not pd.isna(row[col_notes]) else None,
            "shared": str(row[col_shared]).strip() if col_shared and not pd.isna(row[col_shared]) else "1. Geteilt",
            "entry_type": entry_type,
            "reason": str(row[col_reason]).strip() if col_reason and not pd.isna(row[col_reason]) else None,
        }
        entries.append(entry)

    return entries
