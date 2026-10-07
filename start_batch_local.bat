@echo off
cd /d %~dp0
rem Python環境はパス長制限（260文字）を避けるため短い場所に作る。BATID_VENVで変更可
if not defined BATID_VENV set BATID_VENV=%LOCALAPPDATA%\bat_id_japan\venv_batch
if not exist "%BATID_VENV%\Scripts\python.exe" (
  echo 先に setup_windows.bat をダブルクリックしてセットアップしてください。
  pause
  exit /b 1
)
echo コウモリ一括判定アプリを起動しています。まもなくブラウザが開きます。
echo 使い終わったら、この黒い画面を閉じてください。
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8501"
"%BATID_VENV%\Scripts\python.exe" -m streamlit run app_batch_local.py --server.headless true --server.port 8501
pause
