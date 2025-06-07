@echo off
echo ====================================
echo Structural Modeling Toolkit Launcher
echo ====================================
echo.
echo 1. Run Integrated Workflow
echo 2. Run Structural Grid Generator
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    python run_workflow.py
) else if "%choice%"=="2" (
    python run_grid.py
) else if "%choice%"=="3" (
    exit
) else (
    echo Invalid choice. Please try again.
    pause
)
