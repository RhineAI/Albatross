@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
cd /d c:\Projects\Albatross
uv run python one_token.py > c:\Projects\Albatross\run_output.txt 2>&1
echo DONE >> c:\Projects\Albatross\run_output.txt
