@echo off
title Ace-Step Lite Launcher
color 0d

echo ==========================================
echo   Ace-Step Lite - Auto Setup + Launcher
echo ==========================================
echo.

REM ------------------------------------------
REM CHECK PYTHON
REM ------------------------------------------
echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Install Python 3.10 or 3.11 from python.org
    pause
    exit /b
)

REM ------------------------------------------
REM UPGRADE PIP
REM ------------------------------------------
echo Upgrading pip...
python -m pip install --upgrade pip

REM ------------------------------------------
REM CHECK FOR NVIDIA GPU
REM ------------------------------------------
echo Detecting GPU...
python -c "import torch; print('CUDA available:' , torch.cuda.is_available())" > gpucheck.txt

findstr /C:"True" gpucheck.txt >nul
if %errorlevel%==0 (
    echo [GPU] NVIDIA GPU detected. Installing CUDA-enabled PyTorch...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [GPU] No NVIDIA GPU detected. Installing CPU-only PyTorch...
    python -m pip install torch torchvision torchaudio
)

del gpucheck.txt

REM ------------------------------------------
REM INSTALL OTHER DEPENDENCIES
REM ------------------------------------------
echo Installing required Python packages...
python -m pip install transformers accelerate sentencepiece gradio numpy soundfile

echo.
echo ==========================================
echo   All dependencies installed!
echo   Launching Ace-Step Lite...
echo ==========================================
echo.

REM ------------------------------------------
REM RUN THE APP
REM ------------------------------------------
python ace_step_lite.py

echo.
pause
