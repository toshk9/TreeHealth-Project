@echo off

set DATA_URL=https://drive.google.com/drive/folders/1V0UEf9TG8c7UmzBOIboh-UUoiQ0ZUZox?usp=drive_link
set MODEL_URL=https://drive.google.com/drive/folders/1ESgZgkqz9okM9XJnmfFGXaoz3QMnzYI2?usp=drive_link

for /f "tokens=*" %%i in ('git rev-parse --show-toplevel') do set REPO_ROOT=%%i
echo REPO ROOT: %REPO_ROOT%
cd /d %REPO_ROOT%

@REM python -m venv venv
@REM call venv\Scripts\activate.bat

pip install gdown

echo Downloading folder "data"...
gdown --folder %DATA_URL%

echo Downloading folder "model"...
gdown --folder %MODEL_URL%

echo Download completed.
pause
