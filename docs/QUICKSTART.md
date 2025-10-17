# Shift Prompt Studio - Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd ShiftMaster2000

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Quick Workflow

### 1. Setup (5 minutes)

**Employees Tab:**
- Click "➕ New employee"
- Enter: Name, Email (required for Teams), Group
- Select allowed roles
- Save

**Shifts Tab:**
- Click "➕ New shift"
- Enter: Shift ID, Role, Start/End times
- Select Teams color (1-13)
- Set per-weekday headcount
- Save

### 2. Import Existing Schedule (Optional)

**Import Schedule Tab:**
- Upload Teams Shifts Excel file
- Upload Teams Time-off Excel file (optional)
- Click "Parse and Import"
- Review preview

### 3. Configure Generation

**Planning Period Tab:**
- Set start date (e.g., today)
- Set end date (e.g., today + 6 days)

**LLM Settings Tab:**
- Select model family: Claude or OpenAI
- **For Claude**:
  - Choose model (e.g., claude-sonnet-4-5-20250514)
  - Enable extended thinking
  - Set thinking budget (10000 tokens recommended)
- **For OpenAI**:
  - Choose model (e.g., gpt-5)
  - Set reasoning effort (medium or high)

### 4. Generate Schedule

**Generate Tab:**
- Click "🚀 Generate Schedule"
- Wait for generation (streaming shows progress)
- Review reasoning steps
- Check generated JSON

### 5. Review & Export

**Preview Tab:**
- View calendar visualization
- Check statistics
- Review conflicts

**Export Tab:**
- Click "📥 Export to Excel"
- Download shifts file
- Download time-off file
- Import to Microsoft Teams

## Example: First Schedule

### Step 1: Add 3 Employees
```
Employee 1:
- Name: Doe, John-MGB
- Email: john.doe@mgb.ch
- Group: Service Desk
- Roles: Contact Team, Dispatcher

Employee 2:
- Name: Smith, Jane-MGB
- Email: jane.smith@mgb.ch
- Group: Service Desk
- Roles: Contact Team, SOB Wove

Employee 3:
- Name: Brown, Bob-MGB
- Email: bob.brown@mgb.ch
- Group: Service Desk
- Roles: Operation Lead
```

### Step 2: Add 2 Shift Templates
```
Shift 1:
- ID: contact-0700
- Role: Contact Team
- Time: 07:00 - 16:00
- Color: 2. Blau
- Weekdays: Mon-Fri
- Required: 2 per day

Shift 2:
- ID: operation-0700
- Role: Operation Lead
- Time: 07:00 - 16:00
- Color: 1. Weiß
- Weekdays: Mon-Fri
- Required: 1 per day
```

### Step 3: Set Planning Period
```
Start: 2025-10-20
End: 2025-10-24
(5 days, Mon-Fri)
```

### Step 4: Configure LLM
```
Model: claude-sonnet-4-5-20250514
Extended Thinking: Enabled
Budget: 10000 tokens
Temperature: 0.2
```

### Step 5: Generate
Click "Generate Schedule" and wait ~30 seconds.

### Step 6: Export
Download Excel files and import to Teams!

## Tips

### Best Practices
- ✅ Add employee emails (required for Teams)
- ✅ Set realistic constraints (earliest/latest times)
- ✅ Use fairness hints from imported schedules
- ✅ Start with small planning periods (5-7 days)
- ✅ Review conflicts before exporting

### Troubleshooting

**"No roles to select"**
→ Add shift templates first (roles are inferred from shifts)

**"Could not parse schedule entries"**
→ LLM output may not be valid JSON. Try regenerating or adjusting temperature.

**"Import failed"**
→ Check that Excel file has German column headers from Teams export

**"Generation takes too long"**
→ Reduce thinking budget or planning period length

### LLM Model Selection

**Claude Sonnet 4.5** (Recommended)
- Best for: Complex constraints, fairness balancing
- Thinking budget: 10000-20000 tokens
- Cost: ~$0.50 per schedule

**GPT-5** (Alternative)
- Best for: Fast generation, simple schedules
- Reasoning effort: medium or high
- Cost: Varies by model

## Teams Import/Export

### Export from Teams
1. Open Teams → Shifts
2. Select date range
3. Export → Excel
4. Save two files: shifts and time-off

### Import to Teams
1. Open Teams → Shifts
2. Import → Excel
3. Upload shifts file
4. Upload time-off file
5. Confirm import

## Advanced Features

### MCP Servers
Add external tools for LLM to use:
- Filesystem access
- Web search
- Database queries
- Custom APIs

### Concurrent Shifts
Allow employees to work multiple shifts simultaneously:
- Select shift template
- Add to "Can run concurrently with" list

### Conflict Resolution
- Red highlights: Critical conflicts
- Yellow highlights: Warnings
- Review in Preview tab before export

## Support

### Documentation
- `docs/ARCHITECTURE.md` - Project architecture and development guide
- `docs/SOLVER_INTEGRATION.md` - Constraint solver integration details
- `docs/QUICKSTART.md` - This file

### Common Issues
1. **Missing API key**: LLM features require API keys (set in sidebar)
2. **Color mismatch**: Ensure color codes match Teams (1-13, no 7)
3. **Date format**: Teams expects M/D/YYYY (handled automatically)

## Next Steps

1. ✅ Generate your first schedule
2. ✅ Export to Teams and test import
3. ✅ Refine rules and constraints
4. ✅ Experiment with thinking budgets
5. ✅ Integrate with existing workflows

## Version
2.0.0 - January 2025
