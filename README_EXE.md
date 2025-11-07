# 🚀 Building AI Shift Studio as Windows .exe

Quick guide to create a standalone Windows executable.

## 📋 What You Get

A single `.exe` file that:
- ✅ Runs without Python installation
- ✅ Includes your custom icon
- ✅ Opens automatically in browser
- ✅ Contains all dependencies (300-500 MB)
- ✅ Works on any Windows 10/11 PC

## 🏗️ Build on Windows

### Quick Method (Recommended)

1. **Open Command Prompt** in the project folder
2. **Run the build script:**
   ```cmd
   build.bat
   ```
3. **Wait 5-10 minutes** for the build to complete
4. **Find your .exe** in the `dist/` folder

### Manual Method

If the batch script doesn't work:

```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pyinstaller pillow

# Build
pyinstaller AI_Shift_Studio.spec

# Your .exe is in dist/
```

## 📦 Distribution

**Option 1: Simple ZIP**
- Zip the entire `dist/` folder
- Share the ZIP file
- Users extract and run `AI_Shift_Studio.exe`

**Option 2: Professional Installer**
- Use [Inno Setup](https://jrsoftware.org/isdl.php) (free)
- Create an installer that adds Start Menu shortcuts
- Provides uninstaller

## ⚠️ Important Notes

### Windows Only
You **must** build on a Windows machine. The .exe won't work if built on macOS/Linux.

### First Launch
The app takes 10-15 seconds to start the first time (extracting dependencies).

### Firewall
Windows may ask for firewall permission - click "Allow".

### Antivirus
Some antivirus software flags PyInstaller executables. This is a false positive - the app is safe.

### File Size
The .exe will be 300-500 MB because it includes:
- Python runtime
- Streamlit framework
- All dependencies (pandas, openpyxl, requests, etc.)

This is normal and unavoidable for Python apps.

## 🐛 Troubleshooting

**"Python not found"**
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

**"pyinstaller not found"**
```cmd
pip install pyinstaller
```

**Icon not showing**
The icon should already be converted to `.ico` format in `assets/icon.ico`. If missing:
```cmd
python -c "from PIL import Image; img = Image.open('assets/icon.png'); img.save('assets/icon.ico', format='ICO')"
```

**Build fails**
- Make sure you're in the project folder
- Activate the virtual environment: `venv\Scripts\activate`
- Check for error messages in the output
- See detailed guide in `BUILD_EXE.md`

## 📚 More Information

For advanced options and detailed troubleshooting, see **BUILD_EXE.md**.

## ✨ What Users Need

**Nothing!**

Users just:
1. Download the .exe
2. Double-click to run
3. App opens in their browser

No Python, no pip, no installation - just works! 🎉
