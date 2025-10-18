from __future__ import annotations
from typing import Dict, Any, Optional
import json
from datetime import timedelta
from models import Project, TEAMS_COLOR_NAMES

def json_block(data: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

TEAMS_COLOR_SPEC = """
### Microsoft Teams Shift Colors
Use these color codes when formatting output. Each shift should be assigned an appropriate color:

1. Weiß (White) - Operation Lead
2. Blau (Blue) - Contact Team, Dispatcher (07:00-16:00)
3. Grün (Green) - Contact Team, SOB roles, SOB Wove
4. Lila (Purple) - Late shifts (10:00-19:00)
5. Rosa (Pink) - Special assignments (e.g., Techbar)
6. Gelb (Yellow) - Late shifts (09:00-18:00), Dispatcher
8. Dunkelblau (Dark Blue) - Project work (e.g., M-Industrie Projekt)
9. Dunkelgrün (Dark Green) - WoVe, PCV roles
10. Dunkelviolett (Dark Purple) - Pikett (on-call duty)
11. Dunkelrosa (Dark Pink) - People Developer, Stellvertretung
12. Dunkelgelb (Dark Yellow) - Livechat shifts
13. Grau (Gray) - Time-off, holidays, sick leave

**Note:** Color 7 is not used. When assigning shifts, include the color_code (1-13) in your output.
"""

SYSTEM_TEMPLATE = """\
{preamble}

You will produce a fair, rotation-balanced schedule for the planning period that covers all required roles and shifts,
honors hard constraints, and tries to satisfy soft preferences.
Resolve conflicts explicitly and note any rule violations.

{teams_color_spec}

Data dictionary:
- employees[]: people and their capabilities, languages, constraints, time windows, and email addresses
- shifts[]: templates describing role, start/end times, weekdays, headcount, and Teams color codes
- meta: version, project name
{planning_period_context}

Rules (free text may include exceptions, blackouts, public holidays, and rotation heuristics):
{narrative_rules}

## CRITICAL: Output Format for Microsoft Teams Shifts Import

You MUST return a JSON object with this EXACT structure:

```json
{{
  "shifts": [
    {{
      "employee_name": "Lastname, Firstname-MGB",
      "employee_email": "firstname.lastname@mgb.ch",
      "group": "Service Desk",
      "start_date": "10/1/2025",
      "start_time": "07:00",
      "end_date": "10/1/2025",
      "end_time": "16:00",
      "color_code": "1. Weiß",
      "label": "Operation Lead",
      "unpaid_break": null,
      "notes": "",
      "shared": "1. Geteilt"
    }}
  ],
  "notes": "Schedule generation notes, violations, and explanations"
}}
```

**Field Requirements:**
- `employee_name`: Exact full name from employees list (e.g., "Bänninger, Markus-MGB")
- `employee_email`: Business email from employees list
- `group`: Department/team (e.g., "Service Desk")
- `start_date`: M/D/YYYY format (e.g., "10/1/2025")
- `start_time`: HH:MM 24-hour format (e.g., "07:00")
- `end_date`: M/D/YYYY format (usually same as start_date)
- `end_time`: HH:MM 24-hour format (e.g., "16:00")
- `color_code`: MUST be one of: "1. Weiß", "2. Blau", "3. Grün", "4. Lila", "5. Rosa", "6. Gelb", "8. Dunkelblau", "9. Dunkelgrün", "10. Dunkelviolett", "11. Dunkelrosa", "12. Dunkelgelb", "13. Grau"
- `label`: Role/shift name from shifts templates (e.g., "Operation Lead", "Dispatcher Wove")
- `unpaid_break`: Integer minutes or null
- `notes`: Any relevant notes about this specific assignment
- `shared`: MUST be "1. Geteilt" (shared) or "2. Nicht freigegeben" (not shared)

When you must break a rule to cover a critical shift, prefer breaking soft preferences first. If a role cannot be covered,
explain in the top-level `notes` field. Keep total assignments per person proportional to employment percent when possible.

**IMPORTANT**: Output ONLY valid JSON matching this structure. No additional text before or after the JSON.
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

def build_system_prompt(
    project: Project,
    schedule_payload: Optional[Dict[str, Any]] = None,
    today_iso: Optional[str] = None,
    planning_period: Optional[tuple[str, str]] = None
) -> str:
    data = project.as_compact_json()

    # Build planning period context
    planning_context = ""
    if planning_period:
        start_date, end_date = planning_period
        from datetime import datetime
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        num_days = (end - start).days + 1
        weekdays = []
        current = start
        while current <= end:
            weekdays.append(current.strftime("%A"))
            current += timedelta(days=1)

        planning_context = f"""
### Planning Period
Generate schedule for: **{start_date}** to **{end_date}** (inclusive)
Total days: {num_days}
Weekdays included: {', '.join(set(weekdays))}
"""

    sys = SYSTEM_TEMPLATE.format(
        preamble=project.global_rules.preamble.strip(),
        teams_color_spec=TEAMS_COLOR_SPEC,
        narrative_rules=project.global_rules.narrative_rules.strip(),
        ofmt=project.global_rules.output_format_instructions.strip(),
        planning_period_context=planning_context
    )
    compiled = sys + "\n\nData:\n" + json_block(data)

    if schedule_payload:
        addendum = SCHEDULE_ADDENDUM_TEMPLATE.format(
            today=today_iso or "unknown",
            schedule_json=json_block(schedule_payload)
        )
        compiled += "\n\n" + addendum

    return compiled
