import json
import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any

from models import Project, Employee, ShiftTemplate
from utils import save_project, load_project_dict, compile_prompt, call_llm, parse_schedule_to_payload

st.set_page_config(page_title="Shift Prompt Studio", page_icon="🗓️", layout="wide")

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "project" not in st.session_state:
    st.session_state.project = Project()
if "schedule_payload" not in st.session_state:
    st.session_state.schedule_payload = None

project: Project = st.session_state.project

# ---------------------------------------------------------
# Sidebar: Project & LLM
# ---------------------------------------------------------
with st.sidebar:
    st.title("🗓️ Shift Prompt Studio")
    st.caption("Compose bulletproof system prompts for shift planning.")
    st.write("---")

    colA, colB = st.columns([2,1])
    with colA:
        project.name = st.text_input("Project name", value=project.name)
    with colB:
        project.version = st.text_input("Version", value=project.version)

    st.write("### Project file")
    up = st.file_uploader("Load project (.json)", type=["json"], accept_multiple_files=False, key="proj_upload")
    if up is not None:
        try:
            data = json.load(up)
            project = load_project_dict(data)
            st.session_state.project = project
            st.success(f"Loaded project: {project.name}")
        except Exception as e:
            st.error(f"Failed to load project JSON: {e}")

    dl_name = st.text_input("Save as filename", value=f"{project.name.replace(' ','_').lower()}.json")
    if st.button("💾 Save project"):
        try:
            save_project(dl_name, project)
            st.success(f"Saved {dl_name} in current working directory.")
            st.download_button("Download saved file", data=open(dl_name,"rb").read(), file_name=dl_name)
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.write("---")
    st.write("### LLM settings (optional)")
    endpoint = st.text_input("OpenAI-compatible API base", value="https://api.openai.com")
    api_key = st.text_input("API key", type="password", value="")
    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    json_mode = st.toggle("Force JSON response_format", value=True)
    st.caption("Works with OpenAI, OpenRouter, and most OpenAI-compatible servers.")

st.title("Shift Prompt Studio")

tabs = st.tabs(["Employees", "Shifts & Roles", "Rules & Preamble", "Schedule File (Optional)", "Compile & Export", "Test Run (LLM)"])

# ---------------------------------------------------------
# Employees Tab
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Employees")

    if "editing_employee_id" not in st.session_state:
        st.session_state.editing_employee_id = None

    st.markdown("**Select an employee to edit**")
    existing_emp_names = [e.name for e in project.employees]
    selected_emp_name = st.selectbox(
        "Employee",
        options=["➕ New employee"] + existing_emp_names,
        index=(existing_emp_names.index(next((e.name for e in project.employees if e.id == st.session_state.editing_employee_id), existing_emp_names[0])) + 1) if st.session_state.editing_employee_id else 0
    )

    if selected_emp_name == "➕ New employee":
        st.session_state.editing_employee_id = None
        current_employee = None
    else:
        current_employee = next((e for e in project.employees if e.name == selected_emp_name), None)
        st.session_state.editing_employee_id = current_employee.id if current_employee else None

    with st.expander("➕ Add / edit employee", expanded=True):
        emp_name = st.text_input("Name", value=current_employee.name if current_employee else "")
        emp_email = st.text_input("Email", value=current_employee.email if current_employee else "")
        emp_percent = st.number_input(
            "Employment percent",
            min_value=0,
            max_value=200,
            value=current_employee.percent if (current_employee and current_employee.percent is not None) else 100,
            key="emp_percent_input"
        )

        if "role_options" not in st.session_state:
            st.session_state.role_options = []

        inferred_roles = {
            s.role.strip()
            for s in project.shifts
            if isinstance(s.role, str) and s.role.strip()
        } 
        inferred_roles.update(
            r.strip()
            for employee in project.employees
            for r in employee.roles
            if isinstance(r, str) and r.strip()
        )

        if inferred_roles:
            st.session_state.role_options = sorted(inferred_roles)

        available_roles = st.session_state.role_options

        if not available_roles:
            st.info("No roles defined yet. Add roles in the 'Shifts & Roles' tab to populate this list.")

        current_roles = current_employee.roles if current_employee else []
        selected_roles = st.multiselect(
            "Allowed roles",
            options=available_roles,
            default=current_roles,
            placeholder="Select one or more roles",
            key="emp_roles_multiselect"
        )
        languages = st.text_input(
            "Languages (comma-separated)",
            value=", ".join(current_employee.languages) if current_employee else "DE, FR",
            key="emp_languages_input"
        )
        earliest = st.text_input(
            "Earliest start (HH:MM)",
            value=current_employee.earliest_start if current_employee and current_employee.earliest_start else "07:00",
            key="emp_earliest_input"
        )
        latest = st.text_input(
            "Latest end (HH:MM)",
            value=current_employee.latest_end if current_employee and current_employee.latest_end else "19:00",
            key="emp_latest_input"
        )

        weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        st.markdown("**Weekday blockers** (free text per day, leave blank if none):")
        blockers = {}
        cols = st.columns(7)
        existing_blockers = current_employee.weekday_blockers if current_employee else {}
        for i,wd in enumerate(weekdays):
            with cols[i]:
                blockers[wd] = st.text_input(wd, value=existing_blockers.get(wd, ""), key=f"blk_{wd}")

        hard_constraints = st.text_area(
            "Hard constraints (one per line)",
            height=100,
            value="\n".join(current_employee.hard_constraints) if current_employee else "",
            placeholder="No Pikett the week after vacation.\nNo dispatcher on Tue due to school 07:00-12:00.",
            key="emp_hard_constraints"
        )
        soft_preferences = st.text_area(
            "Soft preferences (one per line)",
            height=100,
            value="\n".join(current_employee.soft_preferences) if current_employee else "",
            placeholder="Prefer 08:00-17:00 when possible.\nAvoid closing shifts on Fridays.",
            key="emp_soft_preferences"
        )

        if st.button("Save employee"):
            if not emp_name.strip():
                st.error("Please enter a name.")
            else:
                emp_id = emp_name.lower().replace(" ","-")
                updated_emp = Employee(
                    id=emp_id,
                    name=emp_name,
                    email=emp_email or None,
                    percent=int(emp_percent) if emp_percent else None,
                    roles=selected_roles,
                    languages=[l.strip() for l in languages.split(",") if l.strip()],
                    earliest_start=earliest or None,
                    latest_end=latest or None,
                    weekday_blockers={k:v for k,v in blockers.items() if v.strip()},
                    hard_constraints=[x.strip() for x in hard_constraints.splitlines() if x.strip()],
                    soft_preferences=[x.strip() for x in soft_preferences.splitlines() if x.strip()],
                )

                existing_idx = next((idx for idx, emp in enumerate(project.employees) if emp.id == st.session_state.editing_employee_id), None)
                if existing_idx is None:
                    project.employees.append(updated_emp)
                    st.session_state.editing_employee_id = emp_id
                    st.success(f"Added {emp_name}")
                else:
                    project.employees[existing_idx] = updated_emp
                    st.session_state.editing_employee_id = emp_id
                    st.success(f"Updated {emp_name}")

                # refresh role options cache based on updated data
                st.session_state.role_options = sorted(
                    {
                        r.strip()
                        for emp in project.employees
                        for r in emp.roles
                        if isinstance(r, str) and r.strip()
                    }
                    .union({
                        sh.role.strip()
                        for sh in project.shifts
                        if isinstance(sh.role, str) and sh.role.strip()
                    })
                )

        if current_employee and st.button("Duplicate employee", key="dup_employee"):
            clone = current_employee.model_copy(deep=True)
            clone_id = f"{clone.id}-copy"
            clone.id = clone_id
            clone.name = f"{clone.name} (copy)"
            project.employees.append(clone)
            st.session_state.editing_employee_id = clone_id
            st.success(f"Duplicated {current_employee.name}")

    if project.employees:
        st.markdown("#### Current employees")
        df = pd.DataFrame([e.model_dump() for e in project.employees])
        st.dataframe(df, use_container_width=True)
        to_remove = st.multiselect("Remove employees", [e.name for e in project.employees])
        if st.button("Remove selected"):
            project.employees = [e for e in project.employees if e.name not in to_remove]
            st.success("Removed.")

# ---------------------------------------------------------
# Shifts & Roles Tab
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Shifts & Roles")

    if "editing_shift_id" not in st.session_state:
        st.session_state.editing_shift_id = None

    shift_options = ["➕ New shift"] + [s.id for s in project.shifts]
    selected_shift_id = st.selectbox(
        "Shift",
        options=shift_options,
        index=shift_options.index(st.session_state.editing_shift_id) if st.session_state.editing_shift_id in shift_options else 0,
    )

    if selected_shift_id == "➕ New shift":
        st.session_state.editing_shift_id = None
        current_shift = None
    else:
        current_shift = next((s for s in project.shifts if s.id == selected_shift_id), None)
        st.session_state.editing_shift_id = current_shift.id if current_shift else None

    with st.expander("➕ Add / edit shift template", expanded=True):
        sid = st.text_input("Shift ID", value=current_shift.id if current_shift else "contact-0700", key="shift_id_input")
        role = st.text_input("Role", value=current_shift.role if current_shift else "Contact Team", key="shift_role_input")
        start = st.text_input("Start time (HH:MM)", value=current_shift.start_time if current_shift else "07:00", key="shift_start_input")
        end = st.text_input("End time (HH:MM)", value=current_shift.end_time if current_shift else "16:00", key="shift_end_input")
        weekdays = st.multiselect(
            "Weekdays",
            ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            default=current_shift.weekdays if current_shift else ["Mon","Tue","Wed","Thu","Fri"],
            key="shift_weekdays_multiselect"
        )
        notes = st.text_input("Notes", value=current_shift.notes if current_shift and current_shift.notes else "", key="shift_notes_input")

        st.markdown("**Per-weekday required headcount:**")
        st.caption("Enter headcount per weekday; leave 0 when the shift doesn't run.")
        cols = st.columns(7)
        per = current_shift.required_count.copy() if current_shift else {}
        wds = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        for i,wd in enumerate(wds):
            with cols[i]:
                val = st.number_input(
                    wd,
                    min_value=0,
                    max_value=50,
                    value=per.get(wd, 0),
                    key=f"req_{wd}"
                )
                if val:
                    per[wd] = int(val)
                elif wd in per:
                    del per[wd]

        bulk_value = st.number_input(
            "Set all weekdays to",
            min_value=0,
            max_value=50,
            value=0,
            key="shift_bulk_req"
        )
        if st.button("Apply to all weekdays", key="shift_apply_bulk"):
            for wd in wds:
                st.session_state[f"req_{wd}"] = int(bulk_value)
            st.rerun()

        if st.button("Save shift"):
            if not sid.strip():
                st.error("Shift ID is required.")
            else:
                new_shift = ShiftTemplate(
                    id=sid.strip(),
                    role=role.strip(),
                    start_time=start.strip(),
                    end_time=end.strip(),
                    weekdays=weekdays,
                    required_count=per,
                    notes=notes or None,
                )

                existing_idx = next((idx for idx, sh in enumerate(project.shifts) if sh.id == st.session_state.editing_shift_id), None)
                if existing_idx is None:
                    project.shifts.append(new_shift)
                    st.success(f"Added shift {sid}")
                else:
                    project.shifts[existing_idx] = new_shift
                    st.success(f"Updated shift {sid}")

                st.session_state.editing_shift_id = new_shift.id
                st.session_state.role_options = sorted(
                    {
                        sh.role.strip()
                        for sh in project.shifts
                        if isinstance(sh.role, str) and sh.role.strip()
                    }
                )

    if project.shifts:
        st.markdown("#### Current shifts")
        df = pd.DataFrame([s.model_dump() for s in project.shifts])
        st.dataframe(df, use_container_width=True)
        to_remove = st.multiselect("Remove shifts", [s.id for s in project.shifts])
        if st.button("Remove selected shifts"):
            project.shifts = [s for s in project.shifts if s.id not in to_remove]
            st.session_state.role_options = sorted(
                {
                    sh.role.strip()
                    for sh in project.shifts
                    if isinstance(sh.role, str) and sh.role.strip()
                }
            ) if project.shifts else []
            st.success("Removed shifts.")

# ---------------------------------------------------------
# Rules & Preamble Tab
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Rules & Preamble")
    project.global_rules.preamble = st.text_area("System preamble", value=project.global_rules.preamble, height=120)
    project.global_rules.narrative_rules = st.text_area(
        "Free-text rules and exceptions",
        value=project.global_rules.narrative_rules,
        height=220,
        placeholder=(
            "Beachten: kein Pikett direkt nach Ferien, mindestens eine Woche dazwischen.\n"
            "Dispatcher: mind. 1 FR-sprechende Person pro Woche.\n"
            "Ab Oktober kein SOB Pikett gleichzeitig mit SOB Wove.\n"
            "Meeting-Tag: Marco M. um 08:00, Gabriel um 09:00 (lange Anfahrt)."
        )
    )
    project.global_rules.output_format_instructions = st.text_area(
        "Output format instructions",
        value=project.global_rules.output_format_instructions,
        height=120
    )

# ---------------------------------------------------------
# Schedule File (Optional) Tab
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Optional schedule file")
    st.caption("Upload a single CSV or Excel file that contains both past schedules (history) and future entries (appointments, vacations, etc.).")
    include_schedule = st.toggle("Include this file in the compiled system prompt", value=False)
    sched_file = st.file_uploader("Schedule file (CSV or Excel)", type=["csv","xlsx","xls"], accept_multiple_files=False, key="sched_upload")

    if include_schedule and sched_file is not None:
        try:
            # detect today's date in Europe/Zurich
            today = datetime.now(ZoneInfo("Europe/Zurich")).date()
            payload = parse_schedule_to_payload(sched_file.read(), sched_file.name, today)
            st.session_state.schedule_payload = payload

            st.success("File parsed and split into past vs future.")
            meta = payload["meta"]
            st.json(meta)

            # quick previews
            with st.expander("Fairness hints (last 14 days)"):
                st.json(payload.get("fairness_hints", {}))

            with st.expander("First 20 past entries"):
                st.json(payload["past_entries"][:20])

            with st.expander("First 20 future entries"):
                st.json(payload["future_entries"][:20])

        except Exception as e:
            st.session_state.schedule_payload = None
            st.error(f"Parsing failed: {e}")
    else:
        st.session_state.schedule_payload = None
        st.info("Toggle the switch and upload a file to include it. Otherwise, nothing extra will be added to the prompt.")

# ---------------------------------------------------------
# Compile & Export Tab
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Compile system prompt")
    today_iso = datetime.now(ZoneInfo("Europe/Zurich")).date().isoformat()

    compiled = compile_prompt(
        project,
        schedule_payload=st.session_state.schedule_payload,
        today_iso=today_iso
    )
    st.code(compiled, language="markdown")
    st.download_button("⬇️ Download compiled prompt (.txt)", data=compiled, file_name=f"{project.name.replace(' ','_')}_system_prompt.txt")

# ---------------------------------------------------------
# Test Run (LLM) Tab
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("Optional test run")
    st.caption("Send the compiled system prompt to your chosen model. Provide a short 'user input' that includes timeframe and any extra constraints for this run.")
    user_input = st.text_area("User input", height=140, placeholder="Plan for 2025-10-06..2025-10-12. Teams: Contact/Dispatcher/SOB/...")

    if st.button("▶️ Run with model"):
        if not api_key.strip():
            st.error("Please provide an API key in the sidebar.")
        else:
            prompt = compiled + "\n\nUser input:\n" + user_input
            try:
                out = call_llm(prompt, endpoint=endpoint, api_key=api_key, model=model, temperature=temperature, json_mode=json_mode)
                st.success("Model responded.")
                # Try to detect JSON vs text for nicer display
                try:
                    parsed = json.loads(out)
                    st.json(parsed)
                except Exception:
                    st.code(out, language="json" if json_mode else "markdown")
            except Exception as e:
                st.error(f"LLM call failed: {e}")
