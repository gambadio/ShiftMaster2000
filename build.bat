@echo off
REM AI Shift Studio - Windows Build Script
REM Run this on Windows to create the .exe file

echo ========================================
echo AI Shift Studio - Building Executable
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q pyinstaller pillow pywebview
echo.

REM Clean previous builds
echo Cleaning previous builds...
if exist "build\" rmdir /s /q build
if exist "dist\" rmdir /s /q dist
echo.

REM Build the executable
echo Building executable with PyInstaller...
pyinstaller --clean AI_Shift_Studio.spec
echo.

REM Check if build succeeded
if exist "dist\AI_Shift_Studio.exe" (
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo The single-file executable is located at:
    echo   dist\AI_Shift_Studio.exe
    echo.
    echo This is a STANDALONE executable - just copy and run it!
    echo No need to distribute any other files.
    echo.
    echo Note: First startup may be slower as files are extracted.
    echo.
) else (
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo.
)

pause
