"""
Shift Prompt Studio - Enhanced with Teams Integration
"""

import json
import os
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional
import asyncio

from models import (
    Project, Employee, ShiftTemplate, LLMConfig, MCPServerConfig,
    PlanningPeriod, ScheduleEntry, TEAMS_COLOR_NAMES, ProviderType,
    LLMProviderConfig, ChatSession, ChatMessage, GeneratedScheduleEntry,
    ScheduleConflict, ConflictType, ScheduleState
)
from utils import (
    save_project, save_complete_state, load_project_dict, load_complete_state,
    compile_prompt, parse_schedule_to_payload, parse_dual_schedule_files,
    convert_uploaded_entries_to_schedule_entries, export_schedule_to_teams_excel
)
from schedule_manager import ScheduleManager, parse_llm_schedule_output
from llm_client import create_llm_client, validate_provider_config
from llm_manager import call_llm_with_reasoning, call_llm_sync
from export_teams import export_to_teams_excel, export_to_teams_excel_multisheet, schedule_entries_from_llm_output
from preview import render_calendar_preview, render_statistics, render_conflicts
from mcp_config import format_mcp_tools_for_prompt, get_mcp_server_examples
from prompt_templates import build_system_prompt
from translations import get_text

st.set_page_config(page_title="Shift Prompt Studio", page_icon="🗓️", layout="wide")

# Display autosave restoration banner at the very top
if st.session_state.get("autosave_available"):
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.info(f"💾 **Autosave found** from {st.session_state.get('autosave_time', 'unknown')} - Would you like to restore it?")
    with col2:
        if st.button("✅ Restore", key="restore_autosave"):
            try:
                state = load_complete_state(st.session_state.autosave_data)
                # Restore state
                st.session_state.project = state["project"]
                st.session_state.schedule_manager = ScheduleManager(state["project"])
                if state.get("schedule_manager_state"):
                    st.session_state.schedule_manager.project.schedule_state = state["schedule_manager_state"]
                if state["schedule_payload"]:
                    st.session_state.schedule_payload = state["schedule_payload"]
                if state["generated_schedule"]:
                    st.session_state.generated_schedule = state["generated_schedule"]
                if state["generated_entries"]:
                    st.session_state.generated_entries = state["generated_entries"]
                if state["llm_conversation"]:
                    st.session_state.llm_conversation = state["llm_conversation"]
                st.session_state.autosave_available = False
                st.success("✅ Autosave restored!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to restore autosave: {e}")
    with col3:
        if st.button("❌ Ignore", key="ignore_autosave"):
            st.session_state.autosave_available = False
            st.rerun()

# ---------------------------------------------------------
# Language options compatible with Teams Shifts
# ---------------------------------------------------------
LANGUAGE_OPTIONS = [
    "DE", "FR", "IT", "EN",  # Swiss languages
    "ES", "PT", "NL", "PL", "RU", "AR", "ZH", "JA", "KO",  # Other common languages
    "CS", "DA", "FI", "EL", "HU", "NO", "SV", "TR", "UK",  # European languages
    "HI", "TH", "VI", "ID", "MS", "FA", "HE", "RO", "BG"   # Additional languages
]

# ---------------------------------------------------------
# Custom CSS for better UI
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Minimal spacing adjustments */
    .stSelectbox {
        margin-bottom: 1rem;
    }

    /* Allow tabs to wrap to multiple lines */
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        white-space: nowrap;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------
if "project" not in st.session_state:
    st.session_state.project = Project()
if "schedule_manager" not in st.session_state:
    st.session_state.schedule_manager = ScheduleManager(st.session_state.project)
if "schedule_payload" not in st.session_state:
    st.session_state.schedule_payload = None
if "generated_schedule" not in st.session_state:
    st.session_state.generated_schedule = None
if "generated_entries" not in st.session_state:
    st.session_state.generated_entries = []
if "llm_conversation" not in st.session_state:
    st.session_state.llm_conversation = []
if "streaming_output" not in st.session_state:
    st.session_state.streaming_output = ""
if "thinking_output" not in st.session_state:
    st.session_state.thinking_output = ""
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession()
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "language" not in st.session_state:
    st.session_state.language = "en"
if "autosave_checked" not in st.session_state:
    st.session_state.autosave_checked = False

# Check for autosave file on first load
if not st.session_state.autosave_checked and os.path.exists(".autosave.json"):
    try:
        with open(".autosave.json", "r") as f:
            autosave_data = json.load(f)
            saved_at = autosave_data.get("saved_at", "unknown")
            st.session_state.autosave_data = autosave_data
            st.session_state.autosave_available = True
            st.session_state.autosave_time = saved_at
    except:
        pass
    st.session_state.autosave_checked = True

project: Project = st.session_state.project
lang = st.session_state.language

# Initialize LLM config if not present
if project.llm_config is None:
    project.llm_config = LLMConfig()
if project.llm_config.provider_config is None:
    project.llm_config.provider_config = LLMProviderConfig()

# Initialize planning period if not present
if project.planning_period is None:
    today = date.today()
    project.planning_period = PlanningPeriod(
        start_date=today,
        end_date=today + timedelta(days=6)
    )

# ---------------------------------------------------------
# Sidebar: Project Management
# ---------------------------------------------------------
with st.sidebar:
    st.title(f"🗓️ {get_text('app_title', lang)}")
    st.caption(get_text('app_caption', lang))

    # Language selector
    language_options = {"English": "en", "Deutsch": "de"}
    selected_language = st.selectbox(
        "🌐 Language / Sprache",
        options=list(language_options.keys()),
        index=list(language_options.values()).index(lang)
    )
    if language_options[selected_language] != lang:
        st.session_state.language = language_options[selected_language]
        st.rerun()

    st.write("---")

    colA, colB = st.columns([2,1])
    with colA:
        project.name = st.text_input(get_text('project_name', lang), value=project.name)
    with colB:
        project.version = st.text_input(get_text('version', lang), value=project.version)

    st.write(f"### {get_text('project_file', lang)}")
    up = st.file_uploader(get_text("load_project", lang), type=["json"], key="proj_upload")
    if up is not None:
        try:
            data = json.load(up)
            # Load complete state (backward compatible with old format)
            state = load_complete_state(data)

            # Restore all session state
            st.session_state.project = state["project"]

            # Reinitialize schedule manager with loaded project
            st.session_state.schedule_manager = ScheduleManager(state["project"])

            # Restore schedule manager state if available
            if state.get("schedule_manager_state"):
                st.session_state.schedule_manager.project.schedule_state = state["schedule_manager_state"]

            if state["schedule_payload"]:
                st.session_state.schedule_payload = state["schedule_payload"]
            if state["generated_schedule"]:
                st.session_state.generated_schedule = state["generated_schedule"]
            if state["generated_entries"]:
                st.session_state.generated_entries = state["generated_entries"]
            if state["llm_conversation"]:
                st.session_state.llm_conversation = state["llm_conversation"]

            st.success(f"✅ Loaded: {state['project'].name}")
            entries_count = len(st.session_state.schedule_manager.get_all_entries())
            st.info(f"📊 Restored: {entries_count} schedule entries, {len(state['llm_conversation'])} conversations")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load: {e}")
            st.exception(e)

    dl_name = st.text_input(get_text("save_as", lang), value=f"{project.name.replace(' ','_').lower()}.json")

    # Option to save complete state or just project
    save_mode = st.radio(
        get_text("save_mode", lang),
        [get_text("complete_state", lang), get_text("project_only", lang)],
        help="Complete state saves everything (schedule, results, conversations). Project only saves configuration."
    )

    if st.button(get_text("save_project", lang)):
        try:
            if save_mode == get_text("complete_state", lang):
                # Save complete state including schedule manager
                save_complete_state(
                    dl_name,
                    project,
                    schedule_payload=st.session_state.get("schedule_payload"),
                    generated_schedule=st.session_state.get("generated_schedule"),
                    generated_entries=st.session_state.get("generated_entries", []),
                    llm_conversation=st.session_state.get("llm_conversation", []),
                    schedule_manager_state=st.session_state.schedule_manager.state
                )
                entries_count = len(st.session_state.schedule_manager.get_all_entries())
                st.success(f"✅ Complete state saved to {dl_name} ({entries_count} schedule entries)")
            else:
                save_project(dl_name, project)
                st.success(f"✅ Project config saved to {dl_name}")

            with open(dl_name, "rb") as f:
                st.download_button(get_text("download", lang), data=f.read(), file_name=dl_name, mime="application/json")
        except Exception as e:
            st.error(f"Save failed: {e}")
            st.exception(e)

    st.write("---")

    # Auto-save and Clear State Controls
    st.write(f"### ⚙️ State Management")

    # Auto-save toggle
    auto_save_enabled = st.checkbox(
        "🔄 Auto-save on changes",
        value=st.session_state.get("auto_save_enabled", False),
        help="Automatically save state to .autosave.json when changes are made"
    )
    st.session_state.auto_save_enabled = auto_save_enabled

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Quick Save", help="Save current state to .autosave.json"):
            try:
                save_complete_state(
                    ".autosave.json",
                    project,
                    schedule_payload=st.session_state.get("schedule_payload"),
                    generated_schedule=st.session_state.get("generated_schedule"),
                    generated_entries=st.session_state.get("generated_entries", []),
                    llm_conversation=st.session_state.get("llm_conversation", []),
                    schedule_manager_state=st.session_state.schedule_manager.state
                )
                st.success("✅ Auto-saved")
            except Exception as e:
                st.error(f"Auto-save failed: {e}")

    with col2:
        if st.button("🗑️ Clear All", help="Reset all data (employees, shifts, schedules, etc.)", type="secondary"):
            # Confirm clear
            if st.session_state.get("confirm_clear"):
                st.session_state.project = Project()
                st.session_state.schedule_manager = ScheduleManager(st.session_state.project)
                st.session_state.schedule_payload = None
                st.session_state.generated_schedule = None
                st.session_state.generated_entries = []
                st.session_state.llm_conversation = []
                st.session_state.confirm_clear = False
                st.success("✅ All data cleared")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click 'Clear All' again to confirm")

    st.write("---")
    st.write(f"### {get_text('quick_stats', lang)}")
    st.metric(get_text("employees", lang), len(project.employees))
    st.metric(get_text("shift_templates", lang), len(project.shifts))
    entries_count = len(st.session_state.schedule_manager.get_all_entries())
    st.metric("Schedule Entries", entries_count)
    if project.planning_period:
        days = (project.planning_period.end_date - project.planning_period.start_date).days + 1
        st.metric(get_text("planning_days", lang), days)

# ---------------------------------------------------------
# Tab Structure
# ---------------------------------------------------------
tabs = st.tabs([
    get_text("tab_employees", lang),
    get_text("tab_shifts", lang),
    get_text("tab_rules", lang),
    get_text("tab_import", lang),
    get_text("tab_planning", lang),
    get_text("tab_prompt", lang),
    get_text("tab_llm", lang),
    get_text("tab_chat", lang),
    get_text("tab_generate", lang),
    get_text("tab_preview", lang),
    get_text("tab_export", lang)
])

# ---------------------------------------------------------
# TAB 1: Employees
# ---------------------------------------------------------
with tabs[0]:
    st.subheader(get_text("employee_management", lang))

    if "editing_employee_id" not in st.session_state:
        st.session_state.editing_employee_id = None

    existing_emp_names = [e.name for e in project.employees]

    # Determine current employee first
    if st.session_state.get("selected_emp_name"):
        selected_emp_name = st.session_state.selected_emp_name
    else:
        selected_emp_name = get_text("new_employee", lang)

    if selected_emp_name == get_text("new_employee", lang):
        current_employee = None
    else:
        current_employee = next((e for e in project.employees if e.name == selected_emp_name), None)

    # Put the form ABOVE the selector
    with st.expander(get_text("add_edit_employee", lang), expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            emp_name = st.text_input(get_text("name", lang), value=current_employee.name if current_employee else "")
            emp_email = st.text_input(get_text("email", lang), value=current_employee.email if current_employee else "")
            emp_group = st.text_input(get_text("group_team", lang), value=current_employee.group if current_employee else "Service Desk")

        with col2:
            emp_percent = st.number_input(get_text("employment_percent", lang), 0, 200,
                value=current_employee.percent if (current_employee and current_employee.percent) else 100)

            # Role inference
            if "role_options" not in st.session_state:
                st.session_state.role_options = []

            inferred_roles = {s.role.strip() for s in project.shifts if s.role.strip()}
            inferred_roles.update(r.strip() for e in project.employees for r in e.roles if r.strip())

            if inferred_roles:
                st.session_state.role_options = sorted(inferred_roles)

            current_roles = current_employee.roles if current_employee else []
            selected_roles = st.multiselect(
                get_text("allowed_roles", lang),
                options=st.session_state.role_options,
                default=current_roles,
                key="emp_roles"
            )

        # Language multiselect with predefined options
        current_languages = current_employee.languages if current_employee else ["DE", "FR"]
        selected_languages = st.multiselect(
            get_text("languages", lang),
            options=LANGUAGE_OPTIONS,
            default=current_languages,
            key="emp_languages",
            help="Select one or more languages the employee speaks"
        )

        col3, col4 = st.columns(2)
        with col3:
            earliest = st.text_input(get_text("earliest_start", lang),
                value=current_employee.earliest_start if current_employee and current_employee.earliest_start else "07:00")
        with col4:
            latest = st.text_input(get_text("latest_end", lang),
                value=current_employee.latest_end if current_employee and current_employee.latest_end else "19:00")

        st.markdown(f"**{get_text('weekday_blockers', lang)}**")
        weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        blockers = {}
        cols = st.columns(7)
        existing_blockers = current_employee.weekday_blockers if current_employee else {}
        for i, wd in enumerate(weekdays):
            with cols[i]:
                blockers[wd] = st.text_input(wd, value=existing_blockers.get(wd, ""), key=f"blk_{wd}")

        hard_constraints = st.text_area(get_text("hard_constraints", lang), height=100,
            value="\n".join(current_employee.hard_constraints) if current_employee else "",
            key="emp_hard")

        soft_preferences = st.text_area(get_text("soft_preferences", lang), height=100,
            value="\n".join(current_employee.soft_preferences) if current_employee else "",
            key="emp_soft")

        if st.button(get_text("save_employee", lang)):
            if not emp_name.strip():
                st.error("Name is required")
            elif not emp_email.strip():
                st.error("Email is required for Teams export")
            else:
                emp_id = emp_name.lower().replace(" ","-")
                updated_emp = Employee(
                    id=emp_id,
                    name=emp_name,
                    email=emp_email,
                    group=emp_group or None,
                    percent=int(emp_percent),
                    roles=selected_roles,
                    languages=selected_languages,
                    earliest_start=earliest or None,
                    latest_end=latest or None,
                    weekday_blockers={k:v for k,v in blockers.items() if v.strip()},
                    hard_constraints=[x.strip() for x in hard_constraints.splitlines() if x.strip()],
                    soft_preferences=[x.strip() for x in soft_preferences.splitlines() if x.strip()],
                )

                existing_idx = next((idx for idx, e in enumerate(project.employees) if e.id == emp_id), None)
                if existing_idx is None:
                    project.employees.append(updated_emp)
                    st.success(f"Added {emp_name}")
                else:
                    project.employees[existing_idx] = updated_emp
                    st.success(f"Updated {emp_name}")
                st.session_state.selected_emp_name = emp_name
                st.rerun()

    # Selector comes AFTER the form
    selected_emp_name = st.selectbox(
        get_text("select_employee", lang),
        options=[get_text("new_employee", lang)] + existing_emp_names,
        key="emp_selector",
        index=0 if not st.session_state.get("selected_emp_name") else
              (existing_emp_names.index(st.session_state.selected_emp_name) + 1
               if st.session_state.get("selected_emp_name") in existing_emp_names else 0)
    )

    # Update session state when selector changes
    if selected_emp_name != st.session_state.get("selected_emp_name"):
        st.session_state.selected_emp_name = selected_emp_name
        st.rerun()

    if project.employees:
        st.markdown(f"#### {get_text('current_employees', lang)}")
        df = pd.DataFrame([e.model_dump() for e in project.employees])
        st.dataframe(df, use_container_width=True)

        to_remove = st.multiselect(get_text("remove_employees", lang), [e.name for e in project.employees])
        if st.button(get_text("remove_selected", lang)):
            project.employees = [e for e in project.employees if e.name not in to_remove]
            if st.session_state.get("selected_emp_name") in to_remove:
                st.session_state.selected_emp_name = get_text("new_employee", lang)
            st.success("Removed")
            st.rerun()

# ---------------------------------------------------------
# TAB 2: Shifts & Roles
# ---------------------------------------------------------
with tabs[1]:
    st.subheader(get_text("shifts_roles", lang))

    if "editing_shift_id" not in st.session_state:
        st.session_state.editing_shift_id = None

    shift_options = [get_text("new_shift", lang)] + [s.id for s in project.shifts]

    # Determine current shift first
    if st.session_state.get("selected_shift_id"):
        selected_shift_id = st.session_state.selected_shift_id
    else:
        selected_shift_id = get_text("new_shift", lang)

    if selected_shift_id == get_text("new_shift", lang):
        current_shift = None
    else:
        current_shift = next((s for s in project.shifts if s.id == selected_shift_id), None)

    # Put the form ABOVE the selector
    with st.expander(get_text("add_edit_shift", lang), expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            sid = st.text_input(get_text("shift_id", lang), value=current_shift.id if current_shift else "contact-0700")
            role = st.text_input(get_text("role", lang), value=current_shift.role if current_shift else "Contact Team")
            start = st.text_input(get_text("start_time", lang), value=current_shift.start_time if current_shift else "07:00")
            end = st.text_input(get_text("end_time", lang), value=current_shift.end_time if current_shift else "16:00")

        with col2:
            # Teams color picker
            color_options = [f"{code}. {name}" for code, name in TEAMS_COLOR_NAMES.items()]
            current_color = None
            if current_shift and current_shift.color_code:
                current_color = f"{current_shift.color_code}. {TEAMS_COLOR_NAMES.get(current_shift.color_code, '')}"

            selected_color = st.selectbox(get_text("teams_color", lang), options=color_options,
                index=color_options.index(current_color) if current_color in color_options else 0,
                help="Color coding for Teams Shifts visual display")
            color_code = selected_color.split(".")[0]

        weekdays = st.multiselect(get_text("weekdays", lang), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            default=current_shift.weekdays if current_shift else ["Mon","Tue","Wed","Thu","Fri"])

        # Concurrent shifts
        other_shifts = [s.id for s in project.shifts if s.id != (current_shift.id if current_shift else "")]
        concurrent = st.multiselect(get_text("concurrent_shifts", lang),
            options=other_shifts,
            default=current_shift.concurrent_shifts if current_shift else [])

        st.markdown(f"**{get_text('per_weekday_headcount', lang)}**")
        cols = st.columns(7)
        per = current_shift.required_count.copy() if current_shift else {}
        wds = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        for i, wd in enumerate(wds):
            with cols[i]:
                val = st.number_input(wd, 0, 50, value=per.get(wd, 0), key=f"req_{wd}")
                if val:
                    per[wd] = int(val)
                elif wd in per:
                    del per[wd]

        notes = st.text_input(get_text("notes", lang), value=current_shift.notes if current_shift and current_shift.notes else "")

        if st.button(get_text("save_shift", lang)):
            if not sid.strip() or not role.strip():
                st.error("Shift ID and Role are required")
            else:
                new_shift = ShiftTemplate(
                    id=sid.strip(),
                    role=role.strip(),
                    start_time=start.strip(),
                    end_time=end.strip(),
                    weekdays=weekdays,
                    required_count=per,
                    notes=notes or None,
                    color_code=color_code,
                    concurrent_shifts=concurrent,
                )

                existing_idx = next((idx for idx, sh in enumerate(project.shifts) if sh.id == sid.strip()), None)
                if existing_idx is None:
                    project.shifts.append(new_shift)
                    st.success(f"Added {sid}")
                else:
                    project.shifts[existing_idx] = new_shift
                    st.success(f"Updated {sid}")
                st.session_state.selected_shift_id = sid.strip()
                st.rerun()

    # Selector comes AFTER the form
    selected_shift_id = st.selectbox(
        get_text("select_shift", lang),
        options=shift_options,
        key="shift_selector",
        index=0 if not st.session_state.get("selected_shift_id") else
              (shift_options.index(st.session_state.selected_shift_id)
               if st.session_state.get("selected_shift_id") in shift_options else 0)
    )

    # Update session state when selector changes
    if selected_shift_id != st.session_state.get("selected_shift_id"):
        st.session_state.selected_shift_id = selected_shift_id
        st.rerun()

    if project.shifts:
        st.markdown(f"#### {get_text('current_shifts', lang)}")
        df = pd.DataFrame([s.model_dump() for s in project.shifts])
        st.dataframe(df, use_container_width=True)

        to_remove = st.multiselect(get_text("remove_shifts", lang), [s.id for s in project.shifts])
        if st.button(get_text("remove_selected_shifts", lang)):
            project.shifts = [s for s in project.shifts if s.id not in to_remove]
            if st.session_state.get("selected_shift_id") in to_remove:
                st.session_state.selected_shift_id = get_text("new_shift", lang)
            st.success("Removed")
            st.rerun()

# ---------------------------------------------------------
# TAB 3: Rules & Preamble
# ---------------------------------------------------------
with tabs[2]:
    st.subheader(get_text("rules_preamble", lang))

    project.global_rules.preamble = st.text_area(get_text("system_preamble", lang),
        value=project.global_rules.preamble, height=120)

    project.global_rules.narrative_rules = st.text_area(get_text("narrative_rules", lang),
        value=project.global_rules.narrative_rules, height=220,
        placeholder="Kein Pikett direkt nach Ferien...\nDispatcher: mind. 1 FR-sprechende Person...")

    project.global_rules.output_format_instructions = st.text_area(get_text("output_format", lang),
        value=project.global_rules.output_format_instructions, height=120)

# ---------------------------------------------------------
# TAB 4: Import Schedule
# ---------------------------------------------------------
with tabs[3]:
    st.subheader(get_text("import_teams", lang))
    st.caption("Upload Teams export: either a single multi-sheet Excel file or separate shifts/time-off files")

    # Import mode selection
    import_mode = st.radio(
        get_text("import_mode", lang),
        options=[get_text("single_file", lang), get_text("separate_files", lang)],
        help="Teams exports contain multiple sheets. You can upload the complete file or split files."
    )

    if import_mode == get_text("single_file", lang):
        st.markdown("### Single File Import")
        st.caption("Upload the complete Teams export file (contains Schichten, Arbeitsfreie Zeit, Mitglieder)")

        # Checkbox for shift pattern detection
        auto_detect_shifts = st.checkbox(
            get_text("auto_detect", lang),
            value=False,
            help="Automatically analyze the schedule data to detect and create shift templates based on recurring patterns (time + role + color)"
        )

        teams_file = st.file_uploader(get_text("teams_file", lang), type=["xlsx","xls"], key="teams_multisheet_upload")

        if st.button(get_text("parse_populate", lang), key="parse_multisheet"):
            if teams_file is None:
                st.error("Teams file is required")
            else:
                try:
                    from utils import parse_teams_excel_multisheet, auto_populate_employees_from_members, generate_schedule_preview, detect_shift_patterns_from_schedule

                    today = datetime.now(ZoneInfo("Europe/Zurich")).date()
                    teams_bytes = teams_file.read()

                    # Parse the Excel file
                    payload = parse_teams_excel_multisheet(
                        teams_bytes,
                        teams_file.name,
                        today
                    )

                    st.session_state.schedule_payload = payload

                    # Convert payload to schedule entries and add to schedule manager
                    try:
                        uploaded_entries, members = convert_uploaded_entries_to_schedule_entries(
                            payload,
                            payload.get("members", [])
                        )
                        st.session_state.schedule_manager.add_uploaded_entries(uploaded_entries, members)
                    except Exception as e:
                        st.warning(f"Could not add entries to schedule manager: {e}")

                    # Auto-detect shift patterns if checkbox is enabled
                    shift_detection_result = None
                    if auto_detect_shifts:
                        shift_detection_result = detect_shift_patterns_from_schedule(
                            payload,
                            st.session_state.project
                        )

                    # Auto-populate employees from members data
                    employee_changes = None
                    needs_rerun = False
                    if payload.get("members"):
                        employee_changes = auto_populate_employees_from_members(
                            st.session_state.project,  # Use session_state.project directly
                            payload["members"]
                        )

                        if employee_changes["added_count"] > 0:
                            st.success(f"✅ Added {employee_changes['added_count']} new employee(s)")
                            needs_rerun = True
                        if employee_changes["existing_count"] > 0:
                            st.info(f"ℹ️ {employee_changes['existing_count']} employee(s) already exist")

                    # Generate preview
                    preview = generate_schedule_preview(payload, employee_changes)
                    st.session_state.schedule_preview = preview

                    # Display preview summary
                    st.markdown(f"### {get_text('import_summary', lang)}")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(get_text("past_entries", lang), preview["summary"]["total_past_entries"])
                    with col2:
                        st.metric(get_text("future_shifts", lang), preview["summary"]["future_shifts"])
                    with col3:
                        st.metric(get_text("future_timeoff", lang), preview["summary"]["future_timeoff"])
                    with col4:
                        st.metric("Employees", preview["summary"]["unique_employees_in_schedule"])

                    # Employee changes section
                    if employee_changes:
                        st.markdown(f"### {get_text('employee_changes', lang)}")
                        emp_col1, emp_col2 = st.columns(2)

                        with emp_col1:
                            st.metric(get_text("added", lang), employee_changes["added_count"])
                            if employee_changes["added_employees"]:
                                with st.expander("View added employees"):
                                    added_df = pd.DataFrame(employee_changes["added_employees"])
                                    st.dataframe(added_df[["name", "email"]], use_container_width=True)

                        with emp_col2:
                            st.metric(get_text("already_existed", lang), employee_changes["existing_count"])
                            if employee_changes["existing_employees"]:
                                with st.expander("View existing employees"):
                                    existing_df = pd.DataFrame(employee_changes["existing_employees"])
                                    st.dataframe(existing_df[["name", "email"]], use_container_width=True)

                    # Shift pattern detection results
                    if shift_detection_result:
                        st.markdown(f"### {get_text('shift_detection', lang)}")
                        shift_col1, shift_col2 = st.columns(2)

                        with shift_col1:
                            st.metric(get_text("patterns_detected", lang), shift_detection_result["detected_count"])
                        with shift_col2:
                            st.metric(get_text("new_shifts_added", lang), shift_detection_result["added_count"])

                        if shift_detection_result.get("patterns"):
                            with st.expander("View detected patterns"):
                                patterns_df = pd.DataFrame(shift_detection_result["patterns"])
                                st.dataframe(patterns_df, use_container_width=True)

                        st.success(shift_detection_result["message"])

                    # Preview sections
                    st.markdown(f"### {get_text('schedule_preview', lang)}")

                    # Future shifts preview
                    if preview["sample_future_shifts"]:
                        with st.expander(f"📆 Future Shifts (showing first 10 of {preview['summary']['future_shifts']})"):
                            shifts_df = pd.DataFrame(preview["sample_future_shifts"])
                            # Select relevant columns for display
                            display_cols = [c for c in ["employee", "start_date", "start_time", "end_time", "label", "color_code"] if c in shifts_df.columns]
                            st.dataframe(shifts_df[display_cols], use_container_width=True)

                    # Future time-off preview
                    if preview["sample_future_timeoff"]:
                        with st.expander(f"🏖️ Future Time-Off (showing first 10 of {preview['summary']['future_timeoff']})"):
                            timeoff_df = pd.DataFrame(preview["sample_future_timeoff"])
                            display_cols = [c for c in ["employee", "start_date", "end_date", "reason", "label"] if c in timeoff_df.columns]
                            st.dataframe(timeoff_df[display_cols], use_container_width=True)

                    with st.expander("📋 Metadata"):
                        st.json(payload["meta"])

                    with st.expander("🔄 Fairness Hints (last 14 days)"):
                        st.json(payload.get("fairness_hints", {}))

                    # Force UI refresh if employees were added
                    if needs_rerun:
                        st.rerun()

                except Exception as e:
                    st.error(f"Import failed: {e}")
                    st.exception(e)
                    st.session_state.schedule_payload = None

    else:  # Separate files mode
        st.markdown("### Separate Files Import")
        st.caption("Upload shifts and time-off as separate Excel files")

        col1, col2 = st.columns(2)

        with col1:
            shifts_file = st.file_uploader(get_text("shifts_file", lang), type=["xlsx","xls"], key="shifts_upload")

        with col2:
            timeoff_file = st.file_uploader(get_text("timeoff_file", lang), type=["xlsx","xls"], key="timeoff_upload")

        if st.button(get_text("parse_separate", lang), key="parse_dual"):
            if shifts_file is None:
                st.error("Shifts file is required")
            else:
                try:
                    today = datetime.now(ZoneInfo("Europe/Zurich")).date()
                    shifts_bytes = shifts_file.read()
                    timeoff_bytes = timeoff_file.read() if timeoff_file else None

                    payload = parse_dual_schedule_files(
                        shifts_bytes, shifts_file.name,
                        timeoff_bytes, timeoff_file.name if timeoff_file else None,
                        today
                    )

                    st.session_state.schedule_payload = payload
                    st.success("✅ Schedule imported successfully!")

                    st.json(payload["meta"])

                    with st.expander("Fairness Hints (last 14 days)"):
                        st.json(payload.get("fairness_hints", {}))

                    with st.expander("Preview: Past Entries (first 20)"):
                        st.json(payload["past_entries"][:20])

                    with st.expander("Preview: Future Entries (first 20)"):
                        st.json(payload["future_entries"][:20])

                except Exception as e:
                    st.error(f"Import failed: {e}")
                    st.exception(e)
                    st.session_state.schedule_payload = None

    # Display loaded schedule summary
    if st.session_state.schedule_payload:
        st.markdown("---")
        st.markdown(f"### {get_text('currently_loaded', lang)}")

        if "schedule_preview" in st.session_state and st.session_state.schedule_preview:
            preview = st.session_state.schedule_preview
            summary = preview["summary"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(get_text("future_shifts", lang), summary["future_shifts"])
            with col2:
                st.metric(get_text("future_timeoff", lang), summary["future_timeoff"])
            with col3:
                st.metric("Employees", summary["unique_employees_in_schedule"])

            if preview.get("employee_changes"):
                emp_changes = preview["employee_changes"]
                if emp_changes["added"] > 0 or emp_changes["already_existed"] > 0:
                    st.caption(f"Employees: {emp_changes['added']} added, {emp_changes['already_existed']} already existed")

        st.success("Schedule data loaded and ready for generation")

# ---------------------------------------------------------
# TAB 5: Planning Period
# ---------------------------------------------------------
with tabs[4]:
    st.subheader(get_text("planning_period", lang))
    st.caption(get_text("planning_caption", lang))

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(get_text("start_date", lang),
            value=project.planning_period.start_date if project.planning_period else date.today())

    with col2:
        end_date = st.date_input(get_text("end_date", lang),
            value=project.planning_period.end_date if project.planning_period else date.today() + timedelta(days=6))

    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be before or equal to end date")
        else:
            project.planning_period = PlanningPeriod(start_date=start_date, end_date=end_date)
            days = (end_date - start_date).days + 1
            st.success(f"Planning period set: {days} days from {start_date} to {end_date}")

# ---------------------------------------------------------
# TAB 6: Prompt Preview
# ---------------------------------------------------------
with tabs[5]:
    st.subheader(get_text("prompt_preview", lang))
    st.caption(get_text("prompt_caption", lang))

    # Compile the prompt
    today_iso = datetime.now(ZoneInfo("Europe/Zurich")).date().isoformat()
    planning_tuple = None
    if project.planning_period:
        planning_tuple = (
            project.planning_period.start_date.isoformat(),
            project.planning_period.end_date.isoformat()
        )

    compiled = build_system_prompt(
        project,
        schedule_payload=st.session_state.schedule_payload,
        today_iso=today_iso,
        planning_period=planning_tuple
    )

    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text("characters", lang), f"{len(compiled):,}")
    with col2:
        st.metric(get_text("approx_tokens", lang), f"{len(compiled)//4:,}")
    with col3:
        if st.session_state.schedule_payload:
            st.metric(get_text("schedule_included", lang), "✓ Yes")
        else:
            st.metric(get_text("schedule_included", lang), "✗ No")

    # Display prompt
    st.code(compiled, language="markdown")

    # Download button
    st.download_button(
        get_text("download_prompt", lang),
        data=compiled,
        file_name=f"{project.name.replace(' ','_')}_system_prompt.txt",
        mime="text/plain",
        use_container_width=True
    )

# ---------------------------------------------------------
# TAB 7: LLM Settings
# ---------------------------------------------------------
with tabs[6]:
    st.subheader(get_text("llm_config", lang))
    st.caption(get_text("llm_caption", lang))

    provider_config = project.llm_config.provider_config

    # Provider selection
    provider = st.selectbox(
        get_text("provider", lang),
        options=[p.value for p in ProviderType],
        index=[p.value for p in ProviderType].index(provider_config.provider.value),
        format_func=lambda x: x.title()
    )
    provider_config.provider = ProviderType(provider)

    st.markdown("---")

    # Provider-specific configuration
    if provider_config.provider == ProviderType.OPENAI:
        st.markdown("### OpenAI Configuration")

        provider_config.api_key = st.text_input(
            get_text("api_key", lang),
            value=provider_config.api_key,
            type="password",
            help="Your OpenAI API key (starts with sk-)"
        )

        # Fetch models button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(get_text("fetch_models", lang)):
                if provider_config.api_key:
                    try:
                        temp_client = create_llm_client(project.llm_config)
                        models = temp_client.fetch_models()
                        provider_config.available_models = models
                        st.success(f"Fetched {len(models)} models")
                    except Exception as e:
                        st.error(f"Failed to fetch models: {e}")
                else:
                    st.error("API key required")

        with col2:
            # Model selection
            if provider_config.available_models:
                model_options = provider_config.available_models
            else:
                model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o3-mini"]

            provider_config.model = st.selectbox(
                get_text("model", lang),
                options=model_options,
                index=model_options.index(provider_config.model) if provider_config.model in model_options else 0
            )

        # Reasoning effort for o1/o3 models
        if provider_config.model and any(x in provider_config.model for x in ["o1", "o3", "gpt-5"]):
            project.llm_config.reasoning_effort = st.select_slider(
                get_text("reasoning_effort", lang),
                options=["minimal", "low", "medium", "high"],
                value=project.llm_config.reasoning_effort or "medium",
                help="Controls thinking depth for reasoning models"
            )

    elif provider_config.provider == ProviderType.OPENROUTER:
        st.markdown("### OpenRouter Configuration")

        provider_config.api_key = st.text_input(
            get_text("api_key", lang),
            value=provider_config.api_key,
            type="password",
            help="Your OpenRouter API key (starts with sk-or-v1-)"
        )

        # OpenRouter-specific fields
        provider_config.http_referer = st.text_input(
            "HTTP Referer (optional)",
            value=provider_config.http_referer or "",
            help="Your site URL for rankings on openrouter.ai"
        )

        provider_config.x_title = st.text_input(
            "App Title (optional)",
            value=provider_config.x_title or "",
            help="Your app name for display on openrouter.ai"
        )

        # Model selection with provider prefix
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(get_text("fetch_models", lang)):
                if provider_config.api_key:
                    try:
                        temp_client = create_llm_client(project.llm_config)
                        models = temp_client.fetch_models()
                        provider_config.available_models = models
                        st.success(f"Fetched {len(models)} models")
                    except Exception as e:
                        st.error(f"Failed to fetch models: {e}")
                else:
                    st.error("API key required")

        with col2:
            if provider_config.available_models:
                model_options = provider_config.available_models
            else:
                model_options = [
                    "openai/gpt-4o",
                    "anthropic/claude-3.7-sonnet",
                    "google/gemini-pro",
                    "meta-llama/llama-3.1-405b"
                ]

            provider_config.model = st.selectbox(
                get_text("model", lang),
                options=model_options,
                index=model_options.index(provider_config.model) if provider_config.model in model_options else 0,
                help="Use provider/model format (e.g., openai/gpt-4o)"
            )

    elif provider_config.provider == ProviderType.AZURE:
        st.markdown("### Azure OpenAI Configuration")

        provider_config.api_key = st.text_input(
            get_text("api_key", lang),
            value=provider_config.api_key,
            type="password",
            help="Your Azure OpenAI API key"
        )

        provider_config.azure_endpoint = st.text_input(
            "Azure Endpoint",
            value=provider_config.azure_endpoint or "",
            placeholder="https://YOUR-RESOURCE.openai.azure.com/",
            help="Your Azure OpenAI resource endpoint"
        )

        provider_config.azure_deployment = st.text_input(
            "Deployment Name",
            value=provider_config.azure_deployment or "",
            help="Name of your deployed model (not the model name)"
        )

        provider_config.api_version = st.text_input(
            "API Version",
            value=provider_config.api_version,
            help="Azure API version (e.g., 2024-10-21)"
        )

        provider_config.model = provider_config.azure_deployment or "gpt-4o"

    elif provider_config.provider == ProviderType.CUSTOM:
        st.markdown("### Custom Endpoint Configuration")

        provider_config.api_key = st.text_input(
            "API Key (optional)",
            value=provider_config.api_key,
            type="password",
            help="Some custom endpoints don't require an API key"
        )

        provider_config.base_url = st.text_input(
            "Base URL",
            value=provider_config.base_url or "",
            placeholder="http://localhost:8000/v1",
            help="OpenAI-compatible base URL"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(get_text("fetch_models", lang)):
                if provider_config.base_url:
                    try:
                        temp_client = create_llm_client(project.llm_config)
                        models = temp_client.fetch_models()
                        provider_config.available_models = models
                        st.success(f"Fetched {len(models)} models")
                    except Exception as e:
                        st.warning(f"Could not fetch models: {e}")
                else:
                    st.error("Base URL required")

        with col2:
            if provider_config.available_models:
                model_options = provider_config.available_models
            else:
                model_options = ["custom-model"]

            provider_config.model = st.text_input(
                "Model Name",
                value=provider_config.model,
                help="Model identifier for your custom endpoint"
            )

    st.markdown("---")
    st.markdown(f"### {get_text('generation_params', lang)}")

    col1, col2 = st.columns(2)
    with col1:
        project.llm_config.temperature = st.slider(
            get_text("temperature", lang),
            0.0, 2.0,
            project.llm_config.temperature,
            0.05,
            help="Controls randomness (0=deterministic, 2=very random)"
        )

        project.llm_config.top_p = st.slider(
            get_text("top_p", lang),
            0.0, 1.0,
            project.llm_config.top_p,
            0.05,
            help="Nucleus sampling threshold"
        )

    with col2:
        project.llm_config.max_tokens = st.number_input(
            get_text("max_tokens", lang),
            min_value=100,
            max_value=1000000,
            value=project.llm_config.max_tokens,
            step=100,
            help="Maximum response length (will fall back to model's max if exceeded)"
        )

        project.llm_config.seed = st.number_input(
            get_text("seed", lang),
            min_value=0,
            max_value=999999,
            value=project.llm_config.seed or 0,
            help="For reproducible outputs (0 = random)"
        )
        if project.llm_config.seed == 0:
            project.llm_config.seed = None

    col3, col4 = st.columns(2)
    with col3:
        project.llm_config.frequency_penalty = st.slider(
            get_text("frequency_penalty", lang),
            -2.0, 2.0,
            project.llm_config.frequency_penalty,
            0.1,
            help="Reduce repetition of tokens"
        )

    with col4:
        project.llm_config.presence_penalty = st.slider(
            get_text("presence_penalty", lang),
            -2.0, 2.0,
            project.llm_config.presence_penalty,
            0.1,
            help="Encourage new topics"
        )

    project.llm_config.enable_streaming = st.checkbox(
        get_text("enable_streaming", lang),
        value=project.llm_config.enable_streaming,
        help="Stream responses token by token"
    )

    project.llm_config.json_mode = st.checkbox(
        get_text("json_mode", lang),
        value=project.llm_config.json_mode,
        help="Request JSON-formatted output (if supported by provider)"
    )

    # Reasoning/Thinking parameters
    with st.expander("🧠 Advanced: Reasoning & Thinking Controls", expanded=False):
        st.caption("Configure extended thinking for reasoning-capable models (OpenAI o1/o3, OpenRouter, Claude)")

        # Reasoning effort (OpenAI/OpenRouter)
        reasoning_effort_options = ["None", "low", "medium", "high"]
        current_effort = project.llm_config.reasoning_effort or "None"
        selected_effort = st.selectbox(
            "💭 Reasoning Effort",
            options=reasoning_effort_options,
            index=reasoning_effort_options.index(current_effort) if current_effort in reasoning_effort_options else 0,
            help="How deeply the model thinks:\n- **low**: Quick, lightweight reasoning\n- **medium**: Balanced thinking (recommended)\n- **high**: Deep, thorough reasoning\n\nSupported by: OpenAI o1/o3, OpenRouter models"
        )
        project.llm_config.reasoning_effort = selected_effort if selected_effort != "None" else None

        # Reasoning max tokens (OpenRouter/Claude)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            reasoning_max_tokens = st.number_input(
                "🎯 Max Reasoning Tokens",
                min_value=0,
                max_value=100000,
                value=project.llm_config.reasoning_max_tokens or 0,
                step=1000,
                help="Maximum tokens for reasoning/thinking (0 = unlimited)\n\nSupported by: OpenRouter, Claude"
            )
            project.llm_config.reasoning_max_tokens = reasoning_max_tokens if reasoning_max_tokens > 0 else None

        with col_r2:
            # Budget tokens (Claude extended thinking)
            budget_tokens = st.number_input(
                "🧮 Claude Budget Tokens",
                min_value=0,
                max_value=100000,
                value=project.llm_config.budget_tokens or 0,
                step=1024,
                help="Extended thinking budget for Claude (min 1024)\n\nLeave at 0 for standard mode."
            )
            project.llm_config.budget_tokens = budget_tokens if budget_tokens >= 1024 else None

        # Reasoning exclude (OpenRouter)
        project.llm_config.reasoning_exclude = st.checkbox(
            "🚫 Hide Reasoning from Output",
            value=project.llm_config.reasoning_exclude,
            help="Let the model think internally but don't show the reasoning in the final response (OpenRouter only)"
        )

    # Validate configuration
    st.markdown("---")
    st.markdown(f"### {get_text('config_status', lang)}")

    is_valid, error_msg = validate_provider_config(provider_config)
    if is_valid:
        st.success("✅ Configuration is valid")
        # Try to create client
        try:
            client = create_llm_client(project.llm_config)
            st.session_state.llm_client = client
            st.info("✅ LLM client initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize client: {e}")
            st.session_state.llm_client = None
    else:
        st.error(f"❌ Invalid configuration: {error_msg}")
        st.session_state.llm_client = None

    # MCP Servers
    with st.expander(get_text("mcp_servers", lang)):
        st.caption("Model Context Protocol allows LLM to use external tools")

        if st.button(get_text("add_mcp", lang)):
            project.llm_config.mcp_servers.append(MCPServerConfig(name="", command=""))
            st.rerun()

        for i, server in enumerate(project.llm_config.mcp_servers):
            st.markdown(f"#### MCP Server {i+1}: {server.name or 'Unnamed'}")

            col1, col2 = st.columns([3, 1])
            with col1:
                server.name = st.text_input("Name", value=server.name, key=f"mcp_name_{i}")
            with col2:
                if st.button("🗑️ Remove", key=f"mcp_remove_{i}"):
                    project.llm_config.mcp_servers.pop(i)
                    st.rerun()

            server.command = st.text_input("Command", value=server.command, key=f"mcp_cmd_{i}")
            args_str = st.text_input("Args (space-separated)", value=" ".join(server.args), key=f"mcp_args_{i}")
            server.args = args_str.split() if args_str.strip() else []
            st.markdown("---")

        with st.expander("📚 MCP Server Examples"):
            examples = get_mcp_server_examples()
            for ex in examples:
                st.markdown(f"**{ex['name']}**")
                st.code(f"Command: {ex['command']}\nArgs: {' '.join(ex['args'])}")
                st.caption(ex['description'])

# ---------------------------------------------------------
# TAB 8: Chat Interface
# ---------------------------------------------------------
with tabs[7]:
    st.subheader(get_text("chat_llm", lang))
    st.caption(get_text("chat_caption", lang))

    if not st.session_state.llm_client:
        st.warning("⚠️ Please configure and validate your LLM settings first (LLM Settings tab)")
    else:
        client = st.session_state.llm_client
        session = st.session_state.chat_session

        # Display token usage stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(get_text("prompt_tokens", lang), f"{session.total_prompt_tokens:,}")
        with col2:
            st.metric(get_text("completion_tokens", lang), f"{session.total_completion_tokens:,}")
        with col3:
            if session.total_reasoning_tokens > 0:
                st.metric(get_text("reasoning_tokens", lang), f"{session.total_reasoning_tokens:,}")
            else:
                st.metric(get_text("total_tokens", lang), f"{session.total_prompt_tokens + session.total_completion_tokens:,}")

        st.markdown("---")

        # Display conversation history
        st.markdown(f"### {get_text('conversation', lang)}")

        if not session.messages:
            st.info("Start a conversation to generate or refine your schedule")
        else:
            for i, msg in enumerate(session.messages):
                if msg.role == "system":
                    continue  # Don't display system messages

                with st.chat_message(msg.role):
                    st.markdown(msg.content)

                    # Show reasoning tokens for assistant messages
                    if msg.role == "assistant" and msg.reasoning_tokens:
                        st.caption(f"🧠 Reasoning tokens: {msg.reasoning_tokens:,}")

        # Chat input
        st.markdown("---")

        # Option to include system prompt
        include_system_prompt = st.checkbox(
            get_text("include_system", lang),
            value=len(session.messages) == 0,
            help="For the first message, include the full shift planning context"
        )

        user_input = st.text_area(
            get_text("your_message", lang),
            height=100,
            placeholder="E.g., 'Generate a schedule for next week' or 'Can you swap Alice and Bob on Monday?'",
            key="chat_input"
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            send_button = st.button(get_text("send", lang), type="primary", use_container_width=True)
        with col2:
            clear_button = st.button(get_text("clear_chat", lang), use_container_width=True)
        with col3:
            if st.button(get_text("save_history", lang), use_container_width=True):
                if session.messages:
                    st.session_state.llm_conversation.append({
                        "timestamp": datetime.now().isoformat(),
                        "messages": [m.model_dump() for m in session.messages],
                        "stats": {
                            "prompt_tokens": session.total_prompt_tokens,
                            "completion_tokens": session.total_completion_tokens,
                            "reasoning_tokens": session.total_reasoning_tokens
                        }
                    })
                    st.success("💾 Conversation saved to history")

        if clear_button:
            st.session_state.chat_session = ChatSession()
            st.rerun()

        if send_button and user_input.strip():
            # Prepare system prompt if requested
            system_prompt = None
            if include_system_prompt and not session.messages:
                # Compile the full system prompt
                today_iso = datetime.now(ZoneInfo("Europe/Zurich")).date().isoformat()
                planning_tuple = None
                if project.planning_period:
                    planning_tuple = (
                        project.planning_period.start_date.isoformat(),
                        project.planning_period.end_date.isoformat()
                    )

                system_prompt = build_system_prompt(
                    project,
                    schedule_payload=st.session_state.schedule_payload,
                    today_iso=today_iso,
                    planning_period=planning_tuple
                )

                # Add MCP tools if configured
                if project.llm_config.mcp_servers:
                    mcp_tools_section = format_mcp_tools_for_prompt(project.llm_config.mcp_servers)
                    system_prompt += "\n\n" + mcp_tools_section

            # Send message
            with st.spinner("🤔 Thinking..."):
                try:
                    response_msg = client.chat(
                        user_message=user_input,
                        session=session,
                        system_prompt=system_prompt
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"Chat failed: {e}")
                    st.exception(e)

        # Conversation history
        if st.session_state.llm_conversation:
            with st.expander(f"📚 Conversation History ({len(st.session_state.llm_conversation)} saved)"):
                for idx, conv in enumerate(reversed(st.session_state.llm_conversation)):
                    st.markdown(f"#### Conversation {len(st.session_state.llm_conversation) - idx}")
                    st.caption(f"Saved: {conv['timestamp']}")

                    stats = conv['stats']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prompt", f"{stats['prompt_tokens']:,}")
                    with col2:
                        st.metric("Completion", f"{stats['completion_tokens']:,}")
                    with col3:
                        if stats.get('reasoning_tokens', 0) > 0:
                            st.metric("Reasoning", f"{stats['reasoning_tokens']:,}")

                    with st.expander("View Messages"):
                        for msg in conv['messages']:
                            if msg['role'] != 'system':
                                st.markdown(f"**{msg['role'].title()}:**")
                                st.text(msg['content'][:500] + ("..." if len(msg['content']) > 500 else ""))
                                st.markdown("---")

# ---------------------------------------------------------
# TAB 9: Generate Schedule
# ---------------------------------------------------------
with tabs[8]:
    st.subheader(get_text("generate_schedule", lang))

    if not project.employees:
        st.warning("⚠️ Add employees first")
    elif not project.shifts:
        st.warning("⚠️ Add shift templates first")
    elif not project.planning_period:
        st.warning("⚠️ Set planning period first")
    else:
        st.success(f"✅ Ready to generate schedule for {len(project.employees)} employees, {len(project.shifts)} shift types")

        if st.button(get_text("generate_button", lang), type="primary"):
            with st.spinner("Generating schedule..."):
                # Build prompt
                planning_tuple = (
                    project.planning_period.start_date.isoformat(),
                    project.planning_period.end_date.isoformat()
                )

                today_iso = date.today().isoformat()

                prompt = build_system_prompt(
                    project,
                    schedule_payload=st.session_state.schedule_payload,
                    today_iso=today_iso,
                    planning_period=planning_tuple
                )

                # Add MCP tools if configured
                if project.llm_config.mcp_servers:
                    mcp_tools_section = format_mcp_tools_for_prompt(project.llm_config.mcp_servers)
                    prompt += "\n\n" + mcp_tools_section

                try:
                    # Call LLM
                    result = call_llm_sync(prompt, project.llm_config, "Produce the schedule now.")

                    st.session_state.generated_schedule = result
                    st.success("✅ Schedule generated!")

                    # Display results
                    if result.get("thinking"):
                        with st.expander("🧠 Reasoning / Thinking"):
                            st.text(result["thinking"])

                    st.markdown(f"### {get_text('generated_schedule', lang)}")
                    st.code(result["content"], language="json")

                    # Try to parse and convert to entries
                    try:
                        schedule_json = json.loads(result["content"])
                        st.write(f"🔍 DEBUG: Parsed JSON with {len(schedule_json.get('shifts', []))} shifts and {len(schedule_json.get('time_off', []))} time-off entries")

                        entries, notes = parse_llm_schedule_output(schedule_json)
                        st.write(f"🔍 DEBUG: parse_llm_schedule_output returned {len(entries)} entries")

                        if entries:
                            st.write(f"🔍 DEBUG: First entry - Employee: {entries[0].employee_name}, Date: {entries[0].start_date}")

                        # Check state before adding
                        before_count = len(st.session_state.schedule_manager.state.generated_entries)
                        st.write(f"🔍 DEBUG: Generated entries BEFORE adding: {before_count}")

                        # Add to schedule manager
                        st.session_state.schedule_manager.add_generated_entries(entries)

                        # Check state after adding
                        after_count = len(st.session_state.schedule_manager.state.generated_entries)
                        st.write(f"🔍 DEBUG: Generated entries AFTER adding: {after_count}")

                        # Also keep in session state for backward compatibility
                        st.session_state.generated_entries = entries

                        # Display success
                        st.success(f"✅ Successfully parsed and added {len(entries)} schedule entries!")
                        st.info("📅 Navigate to the **Preview** tab to see the calendar")

                        if notes:
                            with st.expander("📝 Generation Notes"):
                                st.text(notes)
                    except Exception as e:
                        st.warning(f"Could not parse schedule entries: {e}")

                    # Display usage
                    if result.get("usage"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Input tokens", result["usage"].get("input_tokens", 0))
                        with col2:
                            st.metric("Output tokens", result["usage"].get("output_tokens", 0))
                        with col3:
                            total = result["usage"].get("input_tokens", 0) + result["usage"].get("output_tokens", 0)
                            st.metric("Total tokens", total)

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

# ---------------------------------------------------------
# TAB 10: Preview
# ---------------------------------------------------------
with tabs[9]:
    st.subheader(get_text("preview_schedule", lang))

    # Schedule Management Controls
    schedule_mgr = st.session_state.schedule_manager
    all_entries = schedule_mgr.get_all_entries()

    # DEBUG: Check what's in the manager
    st.write(f"🔍 DEBUG Preview Tab - Uploaded entries: {len(schedule_mgr.state.uploaded_entries)}")
    st.write(f"🔍 DEBUG Preview Tab - Generated entries: {len(schedule_mgr.state.generated_entries)}")
    st.write(f"🔍 DEBUG Preview Tab - Total entries: {len(all_entries)}")

    if schedule_mgr.state.generated_entries:
        st.write(f"🔍 DEBUG Preview Tab - First generated entry: {schedule_mgr.state.generated_entries[0].employee_name}, {schedule_mgr.state.generated_entries[0].start_date}")

    # Control Panel
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        if st.button("🗑️ Clear Generated", help="Remove all generated schedule entries"):
            schedule_mgr.clear_generated()
            st.session_state.generated_entries = []
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Uploaded", help="Remove uploaded schedule data"):
            schedule_mgr.clear_uploaded()
            st.session_state.schedule_payload = None
            st.rerun()

    with col3:
        if st.button("🔄 Regenerate", help="Clear generated entries and return to generation tab"):
            schedule_mgr.clear_generated()
            st.session_state.generated_entries = []
            st.info("Generated entries cleared. Go to 'Shift Prompt Studio' tab to regenerate.")

    with col4:
        # Excel Export
        if len(all_entries) > 0:
            if st.button("📥 Export to Excel", help="Export schedule to Teams-compatible Excel file"):
                st.session_state.show_export_dialog = True

    # Excel Export Dialog
    if st.session_state.get("show_export_dialog", False):
        with st.form("export_form"):
            st.markdown("### Export Schedule to Excel")

            col1, col2 = st.columns(2)
            with col1:
                export_start_date = st.date_input("Start Date", value=datetime.now().date())
            with col2:
                export_end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=30)).date())

            col1, col2 = st.columns(2)
            with col1:
                export_btn = st.form_submit_button("📥 Export", type="primary")
            with col2:
                cancel_btn = st.form_submit_button("Cancel")

            if export_btn:
                try:
                    # Export to Excel
                    output_path = f"schedule_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    members = schedule_mgr.state.uploaded_members or []
                    export_schedule_to_teams_excel(
                        all_entries,
                        members,
                        output_path,
                        export_start_date,
                        export_end_date
                    )

                    # Provide download
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Excel File",
                            data=f,
                            file_name=output_path,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    st.success(f"✅ Exported to {output_path}")
                    st.session_state.show_export_dialog = False
                except Exception as e:
                    st.error(f"Export failed: {e}")

            if cancel_btn:
                st.session_state.show_export_dialog = False
                st.rerun()

    # Check what data we have available
    has_imported = schedule_mgr.state.uploaded_entries and len(schedule_mgr.state.uploaded_entries) > 0
    has_generated = schedule_mgr.state.generated_entries and len(schedule_mgr.state.generated_entries) > 0

    if not has_imported and not has_generated:
        st.info("📋 Import a Teams schedule or generate a new schedule to see preview")
    else:
        # Unified Statistics
        st.markdown("### 📊 Schedule Overview")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Entries", len(all_entries))
        with col2:
            uploaded_count = len(schedule_mgr.state.uploaded_entries)
            st.metric("Uploaded", uploaded_count)
        with col3:
            generated_count = len(schedule_mgr.state.generated_entries)
            st.metric("Generated", generated_count)
        with col4:
            shift_count = sum(1 for e in all_entries if e.entry_type == "shift")
            st.metric("Shifts", shift_count)
        with col5:
            timeoff_count = sum(1 for e in all_entries if e.entry_type == "time_off")
            st.metric("Time-Off", timeoff_count)

        st.markdown("---")

        # Calendar View
        st.markdown("### 📅 Calendar View")

        if all_entries:
            # Date range selection for calendar
            try:
                dates = []
                for e in all_entries:
                    try:
                        dt = pd.to_datetime(e.start_date, format='%m/%d/%Y').date()
                        dates.append(dt)
                    except:
                        try:
                            dt = pd.to_datetime(e.start_date).date()
                            dates.append(dt)
                        except:
                            pass

                if dates:
                    min_date = min(dates)
                    max_date = max(dates)

                    # Convert to ScheduleEntry objects for visualization
                    from models import ScheduleEntry

                    schedule_entries = []
                    for e in all_entries:
                        if isinstance(e, ScheduleEntry):
                            schedule_entries.append(e)
                        else:
                            # Convert GeneratedScheduleEntry to ScheduleEntry
                            schedule_entries.append(ScheduleEntry(
                                employee_name=e.employee_name,
                                employee_email=getattr(e, 'employee_email', None),
                                group=getattr(e, 'group', None),
                                start_date=e.start_date,
                                start_time=getattr(e, 'start_time', None),
                                end_date=getattr(e, 'end_date', e.start_date),
                                end_time=getattr(e, 'end_time', None),
                                color_code=getattr(e, 'color_code', None),
                                label=getattr(e, 'label', None),
                                unpaid_break=getattr(e, 'unpaid_break', None),
                                notes=getattr(e, 'notes', None),
                                shared=getattr(e, 'shared', "1. Geteilt"),
                                entry_type=e.entry_type,
                                reason=getattr(e, 'reason', None)
                            ))

                    # Render calendar
                    render_calendar_preview(
                        schedule_entries,
                        min_date,
                        max_date,
                        title="Schedule Calendar"
                    )

                    # Entry Editor
                    with st.expander("✏️ Edit Schedule Entries", expanded=False):
                        st.markdown("#### Edit Individual Entries")

                        # Select entry type to edit
                        col1, col2 = st.columns(2)
                        with col1:
                            edit_source = st.selectbox(
                                "Source",
                                ["Generated", "Uploaded", "All"],
                                key="edit_source_filter"
                            )
                        with col2:
                            edit_type = st.selectbox(
                                "Type",
                                ["All", "Shifts", "Time-Off"],
                                key="edit_type_filter"
                            )

                        # Filter entries based on selection
                        editable_entries = all_entries
                        if edit_source == "Generated":
                            editable_entries = schedule_mgr.state.generated_entries
                        elif edit_source == "Uploaded":
                            editable_entries = schedule_mgr.state.uploaded_entries

                        if edit_type == "Shifts":
                            editable_entries = [e for e in editable_entries if e.entry_type == "shift"]
                        elif edit_type == "Time-Off":
                            editable_entries = [e for e in editable_entries if e.entry_type == "time_off"]

                        if not editable_entries:
                            st.info("No entries match the selected filters")
                        else:
                            # Display entries for editing
                            st.write(f"Found {len(editable_entries)} entries")

                            # Select an entry to edit
                            entry_options = []
                            for e in editable_entries[:50]:  # Limit to first 50 for performance
                                conflict_marker = "⚠️ " if e.has_conflict else ""
                                entry_label = f"{conflict_marker}{e.employee_name} - {e.start_date} {e.start_time} ({e.label or e.entry_type})"
                                entry_options.append((entry_label, e.id))

                            if entry_options:
                                selected_entry_label = st.selectbox(
                                    "Select Entry to Edit",
                                    [label for label, _ in entry_options],
                                    key="select_entry_to_edit"
                                )

                                # Find the selected entry
                                selected_id = next((eid for label, eid in entry_options if label == selected_entry_label), None)
                                if selected_id:
                                    entry = schedule_mgr.get_entry_by_id(selected_id)
                                    if entry:
                                        st.markdown("---")
                                        st.markdown(f"**Editing:** {entry.employee_name} - {entry.label or entry.entry_type}")

                                        with st.form(f"edit_entry_{selected_id}"):
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                new_start_date = st.text_input("Start Date", value=entry.start_date)
                                                new_start_time = st.text_input("Start Time", value=entry.start_time)
                                            with col2:
                                                new_end_date = st.text_input("End Date", value=entry.end_date)
                                                new_end_time = st.text_input("End Time", value=entry.end_time)

                                            col1, col2 = st.columns(2)
                                            with col1:
                                                color_options = ["1. Weiß", "2. Blau", "3. Grün", "4. Lila", "5. Rosa",
                                                               "6. Gelb", "8. Dunkelblau", "9. Dunkelgrün", "10. Dunkelviolett",
                                                               "11. Dunkelrosa", "12. Dunkelgelb", "13. Grau"]
                                                current_color_idx = color_options.index(entry.color_code) if entry.color_code in color_options else 0
                                                new_color = st.selectbox("Color Code", color_options, index=current_color_idx)
                                            with col2:
                                                new_label = st.text_input("Label", value=entry.label or "")

                                            new_notes = st.text_area("Notes", value=entry.notes or "")

                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                save_btn = st.form_submit_button("💾 Save Changes", type="primary")
                                            with col2:
                                                delete_btn = st.form_submit_button("🗑️ Delete Entry", type="secondary")
                                            with col3:
                                                cancel_btn = st.form_submit_button("Cancel")

                                            if save_btn:
                                                updates = {
                                                    "start_date": new_start_date,
                                                    "start_time": new_start_time,
                                                    "end_date": new_end_date,
                                                    "end_time": new_end_time,
                                                    "color_code": new_color,
                                                    "label": new_label,
                                                    "notes": new_notes
                                                }
                                                if schedule_mgr.update_entry(selected_id, updates):
                                                    st.success("✅ Entry updated successfully")
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to update entry")

                                            if delete_btn:
                                                if schedule_mgr.delete_entry(selected_id):
                                                    st.success("✅ Entry deleted")
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to delete entry")

                else:
                    st.warning("Could not parse dates from schedule entries")
            except Exception as e:
                st.error(f"Error rendering calendar: {e}")

# ---------------------------------------------------------
# TAB 11: Export
# ---------------------------------------------------------
with tabs[10]:
    st.subheader(get_text("export_teams", lang))

    # Get schedule manager and all entries (uploaded + generated)
    schedule_mgr = st.session_state.schedule_manager
    all_entries = schedule_mgr.get_all_entries()

    if not all_entries:
        st.info("Import a Teams schedule or generate a new schedule to export")
    else:
        # Show breakdown
        uploaded_count = len(schedule_mgr.state.uploaded_entries)
        generated_count = len(schedule_mgr.state.generated_entries)

        st.write(f"**Ready to export {len(all_entries)} schedule entries**")
        st.caption(f"📥 Uploaded: {uploaded_count} | 🤖 Generated: {generated_count}")

        entries = all_entries

        # Export mode selection
        export_mode = st.radio(
            get_text("export_format", lang),
            options=[get_text("single_file", lang), get_text("separate_files", lang)],
            help="Choose whether to export as one multi-sheet file or separate shift/time-off files"
        )

        if export_mode == get_text("single_file", lang):
            st.markdown(f"### {get_text('single_file_export', lang)}")
            st.caption("Creates one Excel file with sheets: Schichten, Arbeitsfreie Zeit, Mitglieder")

            multisheet_filename = st.text_input(get_text("excel_filename", lang), value="teams_schedule.xlsx")

            if st.button(get_text("export_multisheet", lang), key="export_multisheet"):
                try:
                    # Get members from imported schedule if available
                    members = None
                    if st.session_state.schedule_payload and st.session_state.schedule_payload.get("members"):
                        members = st.session_state.schedule_payload["members"]

                    export_to_teams_excel_multisheet(entries, multisheet_filename, members)
                    st.success(f"✅ Exported to {multisheet_filename}")

                    # Offer download
                    with open(multisheet_filename, "rb") as f:
                        st.download_button(
                            "⬇️ Download Teams Schedule File",
                            data=f.read(),
                            file_name=multisheet_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Export failed: {e}")
                    st.exception(e)

        else:  # Separate files mode
            st.markdown(f"### {get_text('separate_files_export', lang)}")
            st.caption("Creates two separate Excel files for shifts and time-off")

            col1, col2 = st.columns(2)
            with col1:
                shifts_filename = st.text_input(get_text("shifts_filename", lang), value="teams_shifts.xlsx")
            with col2:
                timeoff_filename = st.text_input(get_text("timeoff_filename", lang), value="teams_timeoff.xlsx")

            if st.button(get_text("export_dual", lang), key="export_dual"):
                try:
                    export_to_teams_excel(entries, shifts_filename, timeoff_filename)
                    st.success(f"✅ Exported to {shifts_filename} and {timeoff_filename}")

                    # Offer downloads
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(shifts_filename, "rb") as f:
                            st.download_button(
                                get_text("download_shifts", lang),
                                data=f.read(),
                                file_name=shifts_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    with col2:
                        with open(timeoff_filename, "rb") as f:
                            st.download_button(
                                get_text("download_timeoff", lang),
                                data=f.read(),
                                file_name=timeoff_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"Export failed: {e}")
                    st.exception(e)

        st.markdown("---")
        st.markdown(f"### {get_text('manual_export', lang)}")
        st.caption("Copy the JSON output and process manually if needed")

        if st.session_state.generated_schedule:
            st.code(st.session_state.generated_schedule.get("content", ""), language="json")
