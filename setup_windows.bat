@echo off
cd /d %~dp0
rem Python環境はパス長制限（260文字）を避けるため短い場所に作る。BATID_VENVで変更可
if not defined BATID_VENV set BATID_VENV=%LOCALAPPDATA%\bat_id_japan\venv_batch
echo ============================================================
echo  日本産コウモリ音声 一括判定アプリ Ver.4.3  セットアップ
echo  （初回のみ。ダウンロード約1.5GB、10～30分かかります）
echo ============================================================
echo.

py -3.13 --version >nul 2>nul
if errorlevel 1 (
  echo [エラー] Python 3.13 が見つかりません。
  echo   https://www.python.org/downloads/ から Python 3.13 をインストールしてから、
  echo   もう一度このファイルをダブルクリックしてください。
  echo.
  pause
  exit /b 1
)

if not exist "%BATID_VENV%\Scripts\python.exe" (
  echo Python環境を作成しています...
  py -3.13 -m venv "%BATID_VENV%"
)

echo 必要なライブラリをインストールしています（時間がかかります）...
"%BATID_VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%BATID_VENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [エラー] インストールに失敗しました。画面の内容を開発者にお知らせください。
  pause
  exit /b 1
)

echo.
echo セットアップが完了しました。
echo 今後は start_batch_local.bat をダブルクリックするとアプリが起動します。
pause
