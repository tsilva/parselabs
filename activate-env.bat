@echo off
setlocal

set "EnvDir=.venv"

if not exist "%EnvDir%" (
    echo Creating uv virtual environment...
    uv venv %EnvDir%
    echo Installing dependencies from requirements.txt...
    uv pip install -r requirements.txt
) else (
    echo Using existing virtual environment at %EnvDir%
)

set "ActivateScript=%EnvDir%\Scripts\activate.bat"
if exist "%ActivateScript%" (
    call "%ActivateScript%"
    echo Environment activated
) else (
    echo Could not find activation script in %EnvDir%
    exit /b 1
)
