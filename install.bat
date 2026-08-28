@echo off
echo ========================================================
echo   GIMP 3 Content-Aware Fill - Installer (Windows)
echo   Repo is the plugin - just clone it into plug-ins
echo ========================================================
echo.
echo Recommended (one-liner, keeps you updated):
echo   cd /d "%APPDATA%\GIMP\3.2\plug-ins"
echo   git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git content-aware-fill
echo.
echo Manual fallback - copying current folder...
set SRC=%~dp0
set DST32=%APPDATA%\GIMP\3.2\plug-ins\content-aware-fill
set DST30=%APPDATA%\GIMP\3.0\plug-ins\content-aware-fill
if not exist "%DST32%" mkdir "%DST32%"
copy /Y "%SRC%content-aware-fill.py" "%DST32%\" >nul
xcopy /E /I /Y "%SRC%caf_engine" "%DST32%\caf_engine" >nul
echo [OK] Copied to %DST32%
if not exist "%DST30%" mkdir "%DST30%"
copy /Y "%SRC%content-aware-fill.py" "%DST30%\" >nul
xcopy /E /I /Y "%SRC%caf_engine" "%DST30%\caf_engine" >nul
echo [OK] Copied to %DST30%
echo.
echo Restart GIMP 3 - Edit ^> Content-Aware Fill...
pause
