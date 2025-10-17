# Constraint Solver Setup Guide

## What is the Constraint Solver?

Shift Prompt Studio includes an **optional** constraint programming solver powered by MiniZinc. This feature uses mathematical optimization to generate provably optimal shift schedules.

### When to Use

- **✅ With Solver:** Complex fairness requirements, strict rotation constraints, large teams (20+ employees), need mathematical guarantees
- **✅ Without Solver:** Most typical scheduling scenarios, smaller teams, preference-based assignments

**The app works perfectly without the solver!** The AI-based mode handles most scheduling needs.

## Installation

### Prerequisites

1. **Python Package** (Already included in requirements.txt):
   ```bash
   pip install minizinc
   ```

2. **MiniZinc Binary** (One-time installation):

#### macOS
```bash
# Option 1: Homebrew (recommended)
brew install minizinc

# Option 2: Download installer
# https://www.minizinc.org/software.html
```

#### Windows
1. Download installer from: https://www.minizinc.org/software.html
2. Run the `.msi` installer (includes Gecode and Chuffed solvers)
3. Restart Streamlit app after installation

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get install minizinc

# Arch Linux
sudo pacman -S minizinc

# Or download from: https://www.minizinc.org/software.html
```

### Verification

After installation, restart the Streamlit app and check the **Planning** tab for a green "✅ MiniZinc detected" status indicator.

## How It Works

### Architecture

1. **Fixed Model**: A pre-written MiniZinc model (`shift_schedule.mzn`) defines scheduling logic
2. **JSON Contract**: The LLM sends structured data (employees, shifts, constraints) as JSON
3. **Solver Execution**: MiniZinc solves the optimization problem (15-second timeout)
4. **Result Parsing**: Assignments and violations are returned as JSON

### Key Features

**Hard Constraints** (must be satisfied):
- Coverage requirements (min headcount per shift)
- Employee availability windows
- Skill/language matching
- One shift per employee per day

**Soft Constraints** (minimized via penalties):
- Fairness (balanced workload distribution)
- No consecutive late shifts
- Minimum gap between Pikett assignments
- French-speaking dispatcher requirements

### Solver Backends

The installation includes two high-performance solvers:
- **Chuffed** (default): Excellent for scheduling problems
- **Gecode**: Alternative solver with different search strategies

## Usage

### Enabling Solver Mode

1. Go to the **Planning** tab
2. Check "🔧 Enable Constraint Solver Mode"
3. Configure weights and constraints
4. The system prompt will automatically include the solver tool definition

### Solver Configuration

**Timeout**: 15 seconds default (adjust for larger problems)

**Penalty Weights**: Control trade-offs between objectives
- **Fairness** (3.0): Higher = more balanced shift distribution
- **Late Violation** (2.0): Penalty for consecutive late shifts
- **Pikett Violation** (2.0): Penalty for short Pikett gaps
- **Coverage** (5.0): Penalty for under-staffing

### LLM Integration

When enabled, the LLM sees a new tool: `solve_with_minizinc`

The LLM will:
1. Analyze your scheduling requirements
2. Build the JSON request with all constraints
3. Call the solver tool
4. Interpret results and present the schedule

### Result Statuses

- **OPTIMAL**: Best possible solution found ✅
- **FEASIBLE**: Valid solution found, may not be optimal ⚠️
- **INFEASIBLE**: No valid solution exists (constraints conflict) ❌
- **TIMEOUT**: Solver ran out of time (increase timeout or relax constraints) ⏱️

## Troubleshooting

### "MiniZinc not available"

**Problem**: Python package installed, but binary not found

**Solutions**:
1. Verify installation: `minizinc --version` in terminal
2. Add MiniZinc to PATH (if installed in non-standard location)
3. Restart terminal and Streamlit app

### "No solver available"

**Problem**: MiniZinc installed without solver backends

**Solution**: Download the "bundled" version from minizinc.org (includes Gecode/Chuffed)

### "INFEASIBLE" Results

**Problem**: Solver cannot find any valid schedule

**Common Causes**:
1. **Insufficient staff**: Not enough employees to meet coverage requirements
2. **Conflicting constraints**: E.g., all FR-speakers unavailable when FR-dispatcher required
3. **Over-constrained availability**: Employees' time windows don't cover needed shifts

**Solutions**:
1. Review coverage requirements (reduce if too high)
2. Check employee availability masks
3. Temporarily disable some soft constraints
4. Add more employees or expand availability windows

### "TIMEOUT" Results

**Problem**: Solver couldn't finish within time limit

**Solutions**:
1. Increase timeout in solver settings (30-60 seconds for large problems)
2. Reduce planning horizon (e.g., schedule 4 days instead of 7)
3. Simplify constraints (reduce penalty weights)

## Technical Details

### Model Location
- **Model File**: `shift_schedule.mzn` (in project root)
- **Wrapper**: `solver_service.py`
- **Schemas**: `solver_models.py`

### Performance

**Typical Solve Times** (Intel i7, 16GB RAM):
- 10 employees, 7 days, 5 shifts: 2-8 seconds
- 25 employees, 7 days, 10 shifts: 5-15 seconds
- 50 employees, 14 days, 15 shifts: 15-60 seconds

### Extending the Model

The MiniZinc model (`shift_schedule.mzn`) can be customized:
- Add new constraint types
- Modify objective function weights
- Implement custom fairness metrics

Consult MiniZinc documentation: https://docs.minizinc.dev/

## FAQ

**Q: Is the solver required?**
A: No! The app works great with AI-only mode. Solver is optional for advanced optimization.

**Q: Can I use both AI and solver?**
A: Yes! Enable solver mode and the LLM will intelligently decide when to use it.

**Q: Is it safe?**
A: Yes. The solver runs a fixed, pre-vetted model with your data. No arbitrary code execution.

**Q: Does it work offline?**
A: Once installed, MiniZinc runs locally. No internet required for solving.

**Q: What about licensing?**
A: MiniZinc is open-source (MPL 2.0). Gecode and Chuffed are also open-source.

## Support

**Issues**: https://github.com/YourRepo/ShiftMaster2000/issues
**MiniZinc Docs**: https://docs.minizinc.dev/
**MiniZinc Forum**: https://groups.google.com/g/minizinc

---

**Bottom Line**: The constraint solver is a powerful optional feature. If you're getting great results with AI-only mode, you don't need it. But if you need mathematical optimization guarantees or face complex scheduling puzzles, it's a game-changer! 🚀
