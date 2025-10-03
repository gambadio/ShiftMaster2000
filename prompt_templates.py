from __future__ import annotations
from typing import Dict, Any, Optional
import json
from models import Project

def json_block(data: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

SYSTEM_TEMPLATE = """\
{preamble}

You will produce a fair, rotation-balanced weekly schedule that covers all required roles and shifts,
honors hard constraints, and tries to satisfy soft preferences.
Resolve conflicts explicitly and record any rule violations with a short explanation and a severity score (1-3).

Data dictionary:
- employees[]: people and their capabilities, languages, constraints, and time windows
- shifts[]: templates describing role, start/end times, weekdays, and headcount
- meta: version, project name

Rules (free text may include exceptions, blackouts, public holidays, and rotation heuristics):
{narrative_rules}

Strict output format (JSON only, no prose):
{ofmt}

When you must break a rule to cover a critical shift, prefer breaking soft preferences first. If a role cannot be covered,
propose alternatives in notes[]. Keep total assignments per person proportional to employment percent when possible.
"""

SCHEDULE_ADDENDUM_TEMPLATE = """\
Additional context from an uploaded schedule file has been provided. Use it to enforce rotation fairness
(e.g., avoid consecutive late shifts or Pikett across weeks) and to respect known future unavailability
(vacations, medical appointments, training, etc.). The model date is **{today}** (Europe/Zurich).

Compact schedule JSON (past vs future):
{schedule_json}

Short guidance based on the file:
- Treat past entries as a history signal, not immutable facts to reproduce.
- Treat future entries as hard unavailability unless labeled "tentative".
- If an entry spans a range, assume all days are affected unless a specific weekday filter is present.
"""

def build_system_prompt(project: Project, schedule_payload: Optional[Dict[str, Any]] = None, today_iso: Optional[str] = None) -> str:
    data = project.as_compact_json()
    sys = SYSTEM_TEMPLATE.format(
        preamble=project.global_rules.preamble.strip(),
        narrative_rules=project.global_rules.narrative_rules.strip(),
        ofmt=project.global_rules.output_format_instructions.strip()
    )
    compiled = sys + "\n\nData:\n" + json_block(data)

    if schedule_payload:
        addendum = SCHEDULE_ADDENDUM_TEMPLATE.format(
            today=today_iso or "unknown",
            schedule_json=json_block(schedule_payload)
        )
        compiled += "\n\n" + addendum

    return compiled
