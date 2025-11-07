# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Shift Studio
"""

block_cipher = None

# Collect all Python files
a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app.py', '.'),
        ('models.py', '.'),
        ('utils.py', '.'),
        ('translations.py', '.'),
        ('preview.py', '.'),
        ('schedule_manager.py', '.'),
        ('llm_client.py', '.'),
        ('llm_manager.py', '.'),
        ('export_teams.py', '.'),
        ('mcp_config.py', '.'),
        ('prompt_templates.py', '.'),
        ('assets/icon.png', 'assets'),
        ('assets/icon.ico', 'assets'),
    ],
    hiddenimports=[
        'streamlit',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.scriptrunner.script_runner',
        'streamlit.web',
        'streamlit.web.cli',
        'pandas',
        'openpyxl',
        'requests',
        'pydantic',
        'anthropic',
        'openai',
        'altair',
        'plotly',
        'watchdog',
        'click',
        'tornado',
        'validators',
        'protobuf',
        'pyarrow',
        'tzdata',
        'zoneinfo',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI_Shift_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)
