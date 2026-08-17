@echo off
echo ========================================================
echo   GIMP 3 Content-Aware Fill Plugin Installer (Windows)
echo ========================================================
echo.

set SCRIPT_DIR=%~dp0
set PLUGIN_SRC=%SCRIPT_DIR%content-aware-fill\content-aware-fill.py

if not exist "%PLUGIN_SRC%" (
    echo [ERROR] Could not find content-aware-fill.py in "%SCRIPT_DIR%content-aware-fill"
    pause
    exit /b 1
)

set GIMP32_DIR=%APPDATA%\GIMP\3.2\plug-ins\content-aware-fill
set GIMP30_DIR=%APPDATA%\GIMP\3.0\plug-ins\content-aware-fill

echo Installing for GIMP 3.2...
if not exist "%GIMP32_DIR%" mkdir "%GIMP32_DIR%"
copy /Y "%PLUGIN_SRC%" "%GIMP32_DIR%\content-aware-fill.py" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Installed to %GIMP32_DIR%
) else (
    echo [WARNING] Could not install to GIMP 3.2 folder.
)

echo.
echo Installing for GIMP 3.0...
if not exist "%GIMP30_DIR%" mkdir "%GIMP30_DIR%"
copy /Y "%PLUGIN_SRC%" "%GIMP30_DIR%\content-aware-fill.py" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Installed to %GIMP30_DIR%
) else (
    echo [WARNING] Could not install to GIMP 3.0 folder.
)

echo.
echo ========================================================
echo   Installation Complete!
echo   Restart GIMP 3 and go to: Edit ^> Content-Aware Fill...
echo ========================================================
echo.
pause
