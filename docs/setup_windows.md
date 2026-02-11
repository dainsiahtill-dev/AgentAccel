# Setup on Windows

## Prerequisites
- Windows 10/11
- Python 3.11+
- Git
- ripgrep (`rg`) optional but recommended

## Install
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Initialize
```powershell
accel init --project .
accel doctor --output json
```

## Build Index and Context
```powershell
accel index build --project .
accel context --project . --task "fix headers input issue" --out context_pack.json
```

## Verify
```powershell
accel verify --project . --changed-files src\foo.py src\bar.ts
```
