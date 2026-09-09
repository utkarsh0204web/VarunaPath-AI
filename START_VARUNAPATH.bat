@echo off
title VarunaPath AI
cd /d "%~dp0"
echo Starting VarunaPath AI...
python -m streamlit run app.py
pause
