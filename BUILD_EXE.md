# Building AI Shift Studio as Windows .exe

This guide explains how to create a standalone Windows executable (.exe) for AI Shift Studio.

## Prerequisites

You need a **Windows machine** or **Windows VM** to build the .exe file. Cross-compilation from macOS/Linux is not reliable for Streamlit apps.

### Required Software

1. **Python 3.10+** (Download from [python.org](https://www.python.org/downloads/))
2. **Git** (optional, for cloning the repository)

## Build Instructions

### Step 1: Prepare the Environment

Open Command Prompt or PowerShell and run:

```bash
# Clone the repository (or download as ZIP)
git clone <your-repo-url>
cd ShiftMaster2000

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyInstaller
pip install pyinstaller
```

### Step 2: Build the Executable

Run PyInstaller with the provided spec file:

```bash
pyinstaller AI_Shift_Studio.spec
```

This will create:
- `dist/AI_Shift_Studio.exe` - The standalone executable

### Step 3: Test the Executable

```bash
cd dist
AI_Shift_Studio.exe
```

The app should:
1. Start automatically
2. Open in your default browser
3. Show the AI Shift Studio interface

## Distribution

The `dist/` folder contains everything needed to run the app:
- `AI_Shift_Studio.exe` - The main executable

You can:
1. Zip the entire `dist` folder
2. Share it with users
3. Users just run `AI_Shift_Studio.exe` - no Python installation needed!

## Troubleshooting

### "Module not found" errors
Add missing modules to the `hiddenimports` list in `AI_Shift_Studio.spec`:

```python
hiddenimports=[
    'streamlit',
    'your_missing_module',  # Add here
    ...
]
```

Then rebuild: `pyinstaller AI_Shift_Studio.spec`

### Icon not showing
Make sure `assets/icon.ico` exists. If not, regenerate it:

```bash
python -c "from PIL import Image; img = Image.open('assets/icon.png'); img.save('assets/icon.ico', format='ICO', sizes=[(256,256), (128,128), (64,64), (48,48), (32,32), (16,16)])"
```

### Console window visible
Edit `AI_Shift_Studio.spec` and change:
```python
console=False,  # Hide console window
```

### Large file size
The .exe will be 300-500 MB due to Python + Streamlit + dependencies. This is normal.

To reduce size slightly:
```python
upx=True,  # Already enabled for compression
```

## Advanced Options

### Custom Port
Edit `launcher.py` to change the default port (8501):

```python
os.environ['STREAMLIT_SERVER_PORT'] = '8080'  # Your port
```

### Auto-update Check
Add version checking in `launcher.py` to notify users of updates.

### Installer Creation
Use **Inno Setup** or **NSIS** to create a professional installer:

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Create an installer script
3. Include the .exe and create Start Menu shortcuts

## Notes

- **First launch** may take 10-15 seconds while extracting dependencies
- **Firewall** may ask for permission (allow it)
- **Antivirus** may scan the .exe (this is normal for PyInstaller builds)
- Users need **no Python installation** to run the .exe

## Quick Build Script

Create `build.bat` for easy building:

```batch
@echo off
echo Building AI Shift Studio...
venv\Scripts\activate
pyinstaller --clean AI_Shift_Studio.spec
echo.
echo Build complete! Find the .exe in the dist\ folder
pause
```

Run: `build.bat`
