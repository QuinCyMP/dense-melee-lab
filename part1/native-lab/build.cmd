@echo off
cd /d "%~dp0"
if not exist build mkdir build
set "ZIG_GLOBAL_CACHE_DIR=%~dp0build\zig-cache"
build\toolchain\zig-x86_64-windows-0.16.0\zig.exe c++ -std=c++20 -O3 -fno-fast-math -target x86_64-windows-gnu -mcpu=x86_64_v3 combat_lab.cpp -o build\combat_lab.exe -lws2_32
if errorlevel 1 exit /b %errorlevel%
python build_identity.py record
exit /b %errorlevel%
