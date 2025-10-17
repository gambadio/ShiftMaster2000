"""
Shift Prompt Studio - Enhanced with Teams Integration
"""

import json
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional
import asyncio

from models import (
    Project, Employee, ShiftTemplate, LLMConfig, MCPServerConfig,
    PlanningPeriod, ScheduleEntry, TEAMS_COLOR_NAMES, ProviderType,
    LLMProviderConfig, ChatSession, ChatMessage
)
from utils import (
    save_project, load_project_dict, compile_prompt,
    parse_schedule_to_payload, parse_dual_schedule_files
)
from llm_client import create_llm_client, validate_provider_config
from llm_manager import call_llm_with_reasoning, call_llm_sync
from export_teams import export_to_teams_excel, schedule_entries_from_llm_output
from preview import render_calendar_preview, render_statistics, render_conflicts
from mcp_config import format_mcp_tools_for_prompt, get_mcp_server_examples
from prompt_templates import build_system_prompt

st.set_page_config(page_title="Shift Prompt Studio", page_icon="🗓️", layout="wide")

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

project: Project = st.session_state.project

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
    st.title("🗓️ Shift Prompt Studio")
    st.caption("Microsoft Teams-compatible shift planning with AI")
    st.write("---")

    colA, colB = st.columns([2,1])
    with colA:
        project.name = st.text_input("Project name", value=project.name)
    with colB:
        project.version = st.text_input("Version", value=project.version)

    st.write("### Project file")
    up = st.file_uploader("Load project (.json)", type=["json"], key="proj_upload")
    if up is not None:
        try:
            data = json.load(up)
            project = load_project_dict(data)
            st.session_state.project = project
            st.success(f"Loaded: {project.name}")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    dl_name = st.text_input("Save as", value=f"{project.name.replace(' ','_').lower()}.json")
    if st.button("💾 Save project"):
        try:
            save_project(dl_name, project)
            st.success(f"Saved {dl_name}")
            with open(dl_name, "rb") as f:
                st.download_button("Download", data=f.read(), file_name=dl_name)
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.write("---")
    st.write("### Quick Stats")
    st.metric("Employees", len(project.employees))
    st.metric("Shift Templates", len(project.shifts))
    if project.planning_period:
        days = (project.planning_period.end_date - project.planning_period.start_date).days + 1
        st.metric("Planning Days", days)

# ---------------------------------------------------------
# Tab Structure
# ---------------------------------------------------------
tabs = st.tabs([
    "👥 Employees",
    "🔄 Shifts & Roles",
    "📋 Rules",
    "📤 Import Schedule",
    "📅 Planning Period",
    "📝 Prompt Preview",
    "🤖 LLM Settings",
    "💬 Chat",
    "✨ Generate",
    "👁️ Preview",
    "💾 Export"
])

# ---------------------------------------------------------
# TAB 1: Employees
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Employee Management")

    if "editing_employee_id" not in st.session_state:
        st.session_state.editing_employee_id = None

    existing_emp_names = [e.name for e in project.employees]

    # Determine current employee first
    if st.session_state.get("selected_emp_name"):
        selected_emp_name = st.session_state.selected_emp_name
    else:
        selected_emp_name = "➕ New employee"

    if selected_emp_name == "➕ New employee":
        current_employee = None
    else:
        current_employee = next((e for e in project.employees if e.name == selected_emp_name), None)

    # Put the form ABOVE the selector
    with st.expander("➕ Add / Edit Employee", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            emp_name = st.text_input("Name*", value=current_employee.name if current_employee else "")
            emp_email = st.text_input("Email (for Teams)*", value=current_employee.email if current_employee else "")
            emp_group = st.text_input("Group/Team", value=current_employee.group if current_employee else "Service Desk")

        with col2:
            emp_percent = st.number_input("Employment %", 0, 200,
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
                "Allowed roles",
                options=st.session_state.role_options,
                default=current_roles,
                key="emp_roles"
            )

        # Language multiselect with predefined options
        current_languages = current_employee.languages if current_employee else ["DE", "FR"]
        selected_languages = st.multiselect(
            "Languages",
            options=LANGUAGE_OPTIONS,
            default=current_languages,
            key="emp_languages",
            help="Select one or more languages the employee speaks"
        )

        col3, col4 = st.columns(2)
        with col3:
            earliest = st.text_input("Earliest start (HH:MM)",
                value=current_employee.earliest_start if current_employee and current_employee.earliest_start else "07:00")
        with col4:
            latest = st.text_input("Latest end (HH:MM)",
                value=current_employee.latest_end if current_employee and current_employee.latest_end else "19:00")

        st.markdown("**Weekday blockers**")
        weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        blockers = {}
        cols = st.columns(7)
        existing_blockers = current_employee.weekday_blockers if current_employee else {}
        for i, wd in enumerate(weekdays):
            with cols[i]:
                blockers[wd] = st.text_input(wd, value=existing_blockers.get(wd, ""), key=f"blk_{wd}")

        hard_constraints = st.text_area("Hard constraints (one per line)", height=100,
            value="\n".join(current_employee.hard_constraints) if current_employee else "",
            key="emp_hard")

        soft_preferences = st.text_area("Soft preferences (one per line)", height=100,
            value="\n".join(current_employee.soft_preferences) if current_employee else "",
            key="emp_soft")

        if st.button("💾 Save Employee"):
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
        "Select employee to edit",
        options=["➕ New employee"] + existing_emp_names,
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
        st.markdown("#### Current Employees")
        df = pd.DataFrame([e.model_dump() for e in project.employees])
        st.dataframe(df, use_container_width=True)

        to_remove = st.multiselect("Remove employees", [e.name for e in project.employees])
        if st.button("🗑️ Remove Selected"):
            project.employees = [e for e in project.employees if e.name not in to_remove]
            if st.session_state.get("selected_emp_name") in to_remove:
                st.session_state.selected_emp_name = "➕ New employee"
            st.success("Removed")
            st.rerun()

# ---------------------------------------------------------
# TAB 2: Shifts & Roles
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Shift Templates & Roles")

    if "editing_shift_id" not in st.session_state:
        st.session_state.editing_shift_id = None

    shift_options = ["➕ New shift"] + [s.id for s in project.shifts]

    # Determine current shift first
    if st.session_state.get("selected_shift_id"):
        selected_shift_id = st.session_state.selected_shift_id
    else:
        selected_shift_id = "➕ New shift"

    if selected_shift_id == "➕ New shift":
        current_shift = None
    else:
        current_shift = next((s for s in project.shifts if s.id == selected_shift_id), None)

    # Put the form ABOVE the selector
    with st.expander("➕ Add / Edit Shift Template", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            sid = st.text_input("Shift ID*", value=current_shift.id if current_shift else "contact-0700")
            role = st.text_input("Role*", value=current_shift.role if current_shift else "Contact Team")
            start = st.text_input("Start time (HH:MM)*", value=current_shift.start_time if current_shift else "07:00")
            end = st.text_input("End time (HH:MM)*", value=current_shift.end_time if current_shift else "16:00")

        with col2:
            # Teams color picker
            color_options = [f"{code}. {name}" for code, name in TEAMS_COLOR_NAMES.items()]
            current_color = None
            if current_shift and current_shift.color_code:
                current_color = f"{current_shift.color_code}. {TEAMS_COLOR_NAMES.get(current_shift.color_code, '')}"

            selected_color = st.selectbox("Teams Color", options=color_options,
                index=color_options.index(current_color) if current_color in color_options else 0)
            color_code = selected_color.split(".")[0]

            unpaid_break = st.number_input("Unpaid break (minutes)", 0, 120,
                value=current_shift.unpaid_break_minutes if current_shift and current_shift.unpaid_break_minutes else 0)

            teams_label = st.text_input("Teams label (optional)",
                value=current_shift.teams_label if current_shift else "")

        weekdays = st.multiselect("Weekdays", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            default=current_shift.weekdays if current_shift else ["Mon","Tue","Wed","Thu","Fri"])

        # Concurrent shifts
        other_shifts = [s.id for s in project.shifts if s.id != (current_shift.id if current_shift else "")]
        concurrent = st.multiselect("Can run concurrently with",
            options=other_shifts,
            default=current_shift.concurrent_shifts if current_shift else [])

        st.markdown("**Per-weekday required headcount**")
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

        notes = st.text_input("Notes", value=current_shift.notes if current_shift and current_shift.notes else "")

        if st.button("💾 Save Shift"):
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
                    unpaid_break_minutes=unpaid_break if unpaid_break > 0 else None,
                    teams_label=teams_label or None,
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
        "Select shift to edit",
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
        st.markdown("#### Current Shifts")
        df = pd.DataFrame([s.model_dump() for s in project.shifts])
        st.dataframe(df, use_container_width=True)

        to_remove = st.multiselect("Remove shifts", [s.id for s in project.shifts])
        if st.button("🗑️ Remove Selected Shifts"):
            project.shifts = [s for s in project.shifts if s.id not in to_remove]
            if st.session_state.get("selected_shift_id") in to_remove:
                st.session_state.selected_shift_id = "➕ New shift"
            st.success("Removed")
            st.rerun()

# ---------------------------------------------------------
# TAB 3: Rules & Preamble
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Rules & Preamble")

    project.global_rules.preamble = st.text_area("System preamble",
        value=project.global_rules.preamble, height=120)

    project.global_rules.narrative_rules = st.text_area("Free-text rules and exceptions",
        value=project.global_rules.narrative_rules, height=220,
        placeholder="Kein Pikett direkt nach Ferien...\nDispatcher: mind. 1 FR-sprechende Person...")

    project.global_rules.output_format_instructions = st.text_area("Output format instructions",
        value=project.global_rules.output_format_instructions, height=120)

# ---------------------------------------------------------
# TAB 4: Import Schedule
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Import Microsoft Teams Schedule")
    st.caption("Upload shifts and time-off files exported from Teams")

    col1, col2 = st.columns(2)

    with col1:
        shifts_file = st.file_uploader("Shifts file (Excel)", type=["xlsx","xls"], key="shifts_upload")

    with col2:
        timeoff_file = st.file_uploader("Time-off file (Excel, optional)", type=["xlsx","xls"], key="timeoff_upload")

    if st.button("📥 Parse and Import"):
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
                st.session_state.schedule_payload = None

    if st.session_state.schedule_payload:
        st.info("✅ Schedule data loaded and ready for generation")

# ---------------------------------------------------------
# TAB 5: Planning Period
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Planning Period")
    st.caption("Define the date range for schedule generation")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start date",
            value=project.planning_period.start_date if project.planning_period else date.today())

    with col2:
        end_date = st.date_input("End date",
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
    st.subheader("System Prompt Preview")
    st.caption("Preview the compiled system prompt that will be sent to the LLM")

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
        st.metric("Characters", f"{len(compiled):,}")
    with col2:
        st.metric("Approx. Tokens", f"{len(compiled)//4:,}")
    with col3:
        if st.session_state.schedule_payload:
            st.metric("Schedule Included", "✓ Yes")
        else:
            st.metric("Schedule Included", "✗ No")

    # Display prompt
    st.code(compiled, language="markdown")

    # Download button
    st.download_button(
        "⬇️ Download Prompt (.txt)",
        data=compiled,
        file_name=f"{project.name.replace(' ','_')}_system_prompt.txt",
        mime="text/plain",
        use_container_width=True
    )

# ---------------------------------------------------------
# TAB 7: LLM Settings
# ---------------------------------------------------------
with tabs[6]:
    st.subheader("LLM Provider Configuration")
    st.caption("Configure your LLM provider (OpenAI, OpenRouter, Azure, or Custom)")

    provider_config = project.llm_config.provider_config

    # Provider selection
    provider = st.selectbox(
        "Provider",
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
            "API Key",
            value=provider_config.api_key,
            type="password",
            help="Your OpenAI API key (starts with sk-)"
        )

        # Fetch models button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Fetch Models"):
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
                "Model",
                options=model_options,
                index=model_options.index(provider_config.model) if provider_config.model in model_options else 0
            )

        # Reasoning effort for o1/o3 models
        if provider_config.model and any(x in provider_config.model for x in ["o1", "o3", "gpt-5"]):
            project.llm_config.reasoning_effort = st.select_slider(
                "Reasoning Effort",
                options=["minimal", "low", "medium", "high"],
                value=project.llm_config.reasoning_effort or "medium",
                help="Controls thinking depth for reasoning models"
            )

    elif provider_config.provider == ProviderType.OPENROUTER:
        st.markdown("### OpenRouter Configuration")

        provider_config.api_key = st.text_input(
            "API Key",
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
            if st.button("🔄 Fetch Models"):
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
                "Model",
                options=model_options,
                index=model_options.index(provider_config.model) if provider_config.model in model_options else 0,
                help="Use provider/model format (e.g., openai/gpt-4o)"
            )

    elif provider_config.provider == ProviderType.AZURE:
        st.markdown("### Azure OpenAI Configuration")

        provider_config.api_key = st.text_input(
            "API Key",
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
            if st.button("🔄 Fetch Models"):
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
    st.markdown("### Generation Parameters")

    col1, col2 = st.columns(2)
    with col1:
        project.llm_config.temperature = st.slider(
            "Temperature",
            0.0, 2.0,
            project.llm_config.temperature,
            0.05,
            help="Controls randomness (0=deterministic, 2=very random)"
        )

        project.llm_config.top_p = st.slider(
            "Top P",
            0.0, 1.0,
            project.llm_config.top_p,
            0.05,
            help="Nucleus sampling threshold"
        )

    with col2:
        project.llm_config.max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=32000,
            value=project.llm_config.max_tokens,
            step=100,
            help="Maximum response length"
        )

        project.llm_config.seed = st.number_input(
            "Seed (optional)",
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
            "Frequency Penalty",
            -2.0, 2.0,
            project.llm_config.frequency_penalty,
            0.1,
            help="Reduce repetition of tokens"
        )

    with col4:
        project.llm_config.presence_penalty = st.slider(
            "Presence Penalty",
            -2.0, 2.0,
            project.llm_config.presence_penalty,
            0.1,
            help="Encourage new topics"
        )

    project.llm_config.enable_streaming = st.checkbox(
        "Enable Streaming",
        value=project.llm_config.enable_streaming,
        help="Stream responses token by token"
    )

    project.llm_config.json_mode = st.checkbox(
        "JSON Mode",
        value=project.llm_config.json_mode,
        help="Request JSON-formatted output (if supported by provider)"
    )

    # Validate configuration
    st.markdown("---")
    st.markdown("### Configuration Status")

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
    with st.expander("🔧 MCP Servers (Optional)"):
        st.caption("Model Context Protocol allows LLM to use external tools")

        if st.button("➕ Add MCP Server"):
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
    st.subheader("💬 Chat with LLM")
    st.caption("Interactively refine your schedule using conversation")

    if not st.session_state.llm_client:
        st.warning("⚠️ Please configure and validate your LLM settings first (LLM Settings tab)")
    else:
        client = st.session_state.llm_client
        session = st.session_state.chat_session

        # Display token usage stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prompt Tokens", f"{session.total_prompt_tokens:,}")
        with col2:
            st.metric("Completion Tokens", f"{session.total_completion_tokens:,}")
        with col3:
            if session.total_reasoning_tokens > 0:
                st.metric("Reasoning Tokens", f"{session.total_reasoning_tokens:,}")
            else:
                st.metric("Total Tokens", f"{session.total_prompt_tokens + session.total_completion_tokens:,}")

        st.markdown("---")

        # Display conversation history
        st.markdown("### Conversation")

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
            "Include compiled system prompt",
            value=len(session.messages) == 0,
            help="For the first message, include the full shift planning context"
        )

        user_input = st.text_area(
            "Your message",
            height=100,
            placeholder="E.g., 'Generate a schedule for next week' or 'Can you swap Alice and Bob on Monday?'",
            key="chat_input"
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            send_button = st.button("📤 Send", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
        with col3:
            if st.button("💾 Save to History", use_container_width=True):
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
    st.subheader("Generate Schedule with LLM")

    if not project.employees:
        st.warning("⚠️ Add employees first")
    elif not project.shifts:
        st.warning("⚠️ Add shift templates first")
    elif not project.planning_period:
        st.warning("⚠️ Set planning period first")
    else:
        st.success(f"✅ Ready to generate schedule for {len(project.employees)} employees, {len(project.shifts)} shift types")

        if st.button("🚀 Generate Schedule", type="primary"):
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

                    st.markdown("### Generated Schedule")
                    st.code(result["content"], language="json")

                    # Try to parse and convert to entries
                    try:
                        schedule_json = json.loads(result["content"])
                        entries = schedule_entries_from_llm_output(schedule_json)
                        st.session_state.generated_entries = entries
                        st.info(f"✅ Parsed {len(entries)} schedule entries")
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
    st.subheader("Schedule Preview")

    if not st.session_state.generated_entries:
        st.info("Generate a schedule first to see preview")
    else:
        entries = st.session_state.generated_entries

        # Render statistics
        render_statistics(entries)

        st.markdown("---")

        # Render conflicts
        render_conflicts(entries)

        st.markdown("---")

        # Render calendar
        if project.planning_period:
            render_calendar_preview(
                entries,
                project.planning_period.start_date,
                project.planning_period.end_date,
                title="Generated Schedule Calendar"
            )

# ---------------------------------------------------------
# TAB 11: Export
# ---------------------------------------------------------
with tabs[10]:
    st.subheader("Export to Microsoft Teams")

    if not st.session_state.generated_entries:
        st.info("Generate a schedule first to export")
    else:
        entries = st.session_state.generated_entries

        st.write(f"**Ready to export {len(entries)} schedule entries**")

        shifts_filename = st.text_input("Shifts filename", value="teams_shifts.xlsx")
        timeoff_filename = st.text_input("Time-off filename", value="teams_timeoff.xlsx")

        if st.button("📥 Export to Excel"):
            try:
                export_to_teams_excel(entries, shifts_filename, timeoff_filename)
                st.success(f"✅ Exported to {shifts_filename} and {timeoff_filename}")

                # Offer downloads
                with open(shifts_filename, "rb") as f:
                    st.download_button(
                        "⬇️ Download Shifts File",
                        data=f.read(),
                        file_name=shifts_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                with open(timeoff_filename, "rb") as f:
                    st.download_button(
                        "⬇️ Download Time-off File",
                        data=f.read(),
                        file_name=timeoff_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                st.error(f"Export failed: {e}")
                st.exception(e)

        st.markdown("---")
        st.markdown("### Manual Export")
        st.caption("Copy the JSON output and process manually if needed")

        if st.session_state.generated_schedule:
            st.code(st.session_state.generated_schedule.get("content", ""), language="json")
