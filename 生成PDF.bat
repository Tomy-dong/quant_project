@echo off
cd /d "%~dp0"

set "SCRIPT=..\.trae\skills\md-latex-to-pdf\scripts\build_pdf.py"
set "MD=docs\strategy.md"
set "PDF=docs\strategy.pdf"

echo ==========================================
echo   Building PDF: docs\strategy.pdf
echo ==========================================
echo.

if not exist "%SCRIPT%" goto no_script
if not exist "%MD%" goto no_md

python "%SCRIPT%" --md "%MD%" --pdf "%PDF%"
if not errorlevel 1 goto ok

echo.
echo [WARN] First attempt failed, retrying with longer wait...
python "%SCRIPT%" --md "%MD%" --pdf "%PDF%" --virtual-time-budget 90000
if not errorlevel 1 goto ok
goto failed

:ok
echo.
echo [OK] docs\strategy.pdf generated.
goto end

:no_script
echo [ERROR] script not found: %SCRIPT%
echo Please make sure .trae\skills\md-latex-to-pdf exists.
goto end

:no_md
echo [ERROR] markdown not found: %MD%
goto end

:failed
echo.
echo [FAILED] Formulas did not render. Check network / Edge, then run again.

:end
echo.
pause
