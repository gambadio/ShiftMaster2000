#!/usr/bin/env python3
"""
Quick test to verify comprehensive save/load functionality
"""

import json
from models import Project, Employee, ShiftTemplate, PlanningPeriod, LLMConfig
from utils import save_complete_state, load_complete_state
from datetime import date, timedelta

def test_comprehensive_save_load():
    """Test that all data is saved and loaded correctly"""

    # Create a test project with various data
    project = Project(
        name="Test Project",
        version="3.0"
    )

    # Add an employee
    emp = Employee(
        id="emp_001",
        name="Test Employee",
        email="test@example.com",
        roles=["Support", "Admin"],
        languages=["DE", "EN"]
    )
    project.employees.append(emp)

    # Add a shift
    shift = ShiftTemplate(
        id="morning_shift",
        role="Support",
        start_time="08:00",
        end_time="17:00",
        weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        color_code="2"
    )
    project.shifts.append(shift)

    # Add planning period
    today = date.today()
    project.planning_period = PlanningPeriod(
        start_date=today,
        end_date=today + timedelta(days=7)
    )

    # Add LLM config
    project.llm_config = LLMConfig()

    # Test data
    schedule_payload = {
        "meta": {"test": "data"},
        "past_entries": [],
        "future_entries": [],
        "fairness_hints": {},
        "members": [{"name": "Test Employee", "email": "test@example.com"}]
    }

    generated_schedule = {
        "content": '{"shifts": [], "time_off": []}',
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }

    language = "de"
    last_generation_notes = "Test notes"

    # Save complete state
    test_file = "test_save_comprehensive.json"
    save_complete_state(
        test_file,
        project,
        schedule_payload=schedule_payload,
        generated_schedule=generated_schedule,
        language=language,
        last_generation_notes=last_generation_notes
    )

    print(f"✅ Saved to {test_file}")

    # Load complete state
    with open(test_file, "r") as f:
        data = json.load(f)

    state = load_complete_state(data)

    # Verify all data was saved and loaded
    assert state["project"].name == "Test Project"
    assert len(state["project"].employees) == 1
    assert state["project"].employees[0].name == "Test Employee"
    assert len(state["project"].shifts) == 1
    assert state["project"].shifts[0].id == "morning_shift"
    assert state["project"].planning_period is not None
    assert state["schedule_payload"]["meta"]["test"] == "data"
    assert state["generated_schedule"]["usage"]["input_tokens"] == 100
    assert state["language"] == "de"
    assert state["last_generation_notes"] == "Test notes"

    print("✅ All assertions passed!")
    print("\n📊 Loaded state contains:")
    print(f"  - Project: {state['project'].name}")
    print(f"  - Employees: {len(state['project'].employees)}")
    print(f"  - Shifts: {len(state['project'].shifts)}")
    print(f"  - Planning period: {state['project'].planning_period.start_date} to {state['project'].planning_period.end_date}")
    print(f"  - Schedule payload: {len(state['schedule_payload']['members'])} members")
    print(f"  - Language: {state['language']}")
    print(f"  - Generation notes: {state['last_generation_notes']}")

    # Cleanup
    import os
    os.remove(test_file)
    print(f"\n🧹 Cleaned up {test_file}")

if __name__ == "__main__":
    test_comprehensive_save_load()
