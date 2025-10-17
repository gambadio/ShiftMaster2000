"""
Utility functions for MiniZinc solver integration.

Provides detection, validation, and graceful fallback when MiniZinc is not available.
"""

import sys
from typing import Optional, Tuple
from pathlib import Path


def check_minizinc_available() -> Tuple[bool, str]:
    """
    Check if MiniZinc is available on the system.

    Returns:
        (is_available, message) tuple
        - is_available: True if MiniZinc can be used
        - message: Status message for user display
    """
    try:
        import minizinc

        # Try to find the minizinc driver
        try:
            driver = minizinc.Driver.find()
            version = driver.minizinc_version
            return True, f"✅ MiniZinc {version} detected"
        except Exception as e:
            return False, f"❌ MiniZinc Python package installed, but minizinc binary not found: {e}"

    except ImportError:
        return False, "❌ minizinc Python package not installed (run: pip install minizinc)"


def get_available_solvers() -> list[str]:
    """
    Get list of available solver backends.

    Returns:
        List of solver names (e.g., ["gecode", "chuffed"])
    """
    try:
        import minizinc
        driver = minizinc.Driver.find()

        # Get all available solver configurations
        solvers = []
        for solver_name in ["gecode", "chuffed", "coin-bc", "or-tools"]:
            try:
                solver = minizinc.Solver.lookup(solver_name, driver=driver)
                if solver:
                    solvers.append(solver_name)
            except Exception:
                pass  # Solver not available

        return solvers
    except Exception:
        return []


def get_solver_info() -> dict:
    """
    Get detailed information about MiniZinc installation.

    Returns:
        Dictionary with:
        - available: bool
        - version: str or None
        - solvers: list of available solver names
        - message: status message
    """
    is_available, message = check_minizinc_available()

    if not is_available:
        return {
            "available": False,
            "version": None,
            "solvers": [],
            "message": message
        }

    try:
        import minizinc
        driver = minizinc.Driver.find()
        version = str(driver.minizinc_version)
        solvers = get_available_solvers()

        return {
            "available": True,
            "version": version,
            "solvers": solvers,
            "message": f"✅ MiniZinc {version} with {len(solvers)} solver(s): {', '.join(solvers)}"
        }
    except Exception as e:
        return {
            "available": False,
            "version": None,
            "solvers": [],
            "message": f"❌ Error detecting MiniZinc: {e}"
        }


def get_installation_instructions() -> str:
    """
    Get platform-specific installation instructions for MiniZinc.

    Returns:
        Formatted markdown instructions
    """
    platform = sys.platform

    instructions = """
## Installing MiniZinc (Optional - Enables Constraint Solver)

MiniZinc provides advanced constraint optimization for shift scheduling.

### Quick Install:

"""

    if platform == "darwin":  # macOS
        instructions += """
**macOS:**
```bash
# Option 1: Homebrew (recommended)
brew install minizinc

# Option 2: Download installer
# Visit: https://www.minizinc.org/software.html
```
"""
    elif platform == "win32":  # Windows
        instructions += """
**Windows:**
1. Download the installer from: https://www.minizinc.org/software.html
2. Run the installer (bundled with Gecode and Chuffed solvers)
3. Restart Streamlit app
"""
    else:  # Linux
        instructions += """
**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install minizinc

# Arch Linux
sudo pacman -S minizinc

# Or download from: https://www.minizinc.org/software.html
```
"""

    instructions += """
### After Installation:

1. Install Python package (if not already):
   ```bash
   pip install minizinc
   ```

2. Restart this Streamlit app

3. Enable "Constraint Solver Mode" in the Planning tab

### What You Get:

- ✅ Optimal shift assignments (mathematically proven best solution)
- ✅ Automatic fairness balancing
- ✅ Complex constraint handling (consecutive shifts, rotation patterns)
- ✅ Infeasibility analysis (tells you why a schedule is impossible)

### Alternative: Use LLM-Only Mode

The app works perfectly without MiniZinc using AI-based schedule generation.
The constraint solver is an optional enhancement for complex optimization scenarios.
"""

    return instructions


def validate_solver_request(request_dict: dict) -> Tuple[bool, str]:
    """
    Validate a solver request before execution.

    Args:
        request_dict: Raw request dictionary

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        from solver_models import SolverRequest

        # Try to parse with Pydantic
        SolverRequest.model_validate(request_dict)
        return True, ""

    except Exception as e:
        return False, f"Invalid solver request: {e}"


# Graceful fallback message for UI
SOLVER_UNAVAILABLE_MESSAGE = """
🔧 **Constraint Solver Mode Unavailable**

The constraint solver is an optional feature that requires MiniZinc to be installed.

**Current mode:** AI-based schedule generation (works great for most scenarios)

**To enable constraint solver:**
1. Install MiniZinc from https://www.minizinc.org/software.html
2. Restart this app
3. The solver mode will automatically activate

**Benefits of constraint solver:**
- Provably optimal solutions
- Complex fairness optimization
- Infeasibility analysis
- Mathematical guarantees

**Your app works perfectly without it!** The AI mode handles most scheduling needs.
"""
