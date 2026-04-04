@echo off
set "PYTHON=%APPDATA%\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe"
set "SITE_PACKAGES=%~dp0.venv\Lib\site-packages"
set "PYTHONPATH=src;%SITE_PACKAGES%"
"%PYTHON%" %*
