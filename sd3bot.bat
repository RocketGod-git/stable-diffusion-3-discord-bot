@echo off

REM Check if the virtual environment exists
if not exist "venv" (
    REM Create a new virtual environment
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install or upgrade the required packages (excluding tkinter and transformers)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install Pillow accelerate huggingface_hub sentencepiece protobuf

REM Check if the diffusers directory exists
if not exist "diffusers" (
    REM Install diffusers from source
    pip install git+https://github.com/huggingface/diffusers.git
) else (
    echo The diffusers repository has already been cloned. Skipping installation.
)

pip install transformers discord

REM Run the sd3bot.py script
python sd3bot.py

REM Deactivate the virtual environment
call venv\Scripts\deactivate.bat