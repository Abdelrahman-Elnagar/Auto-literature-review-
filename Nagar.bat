@echo off
echo Starting the Related Work Automation Process...

echo Step 0: Installing required dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies. Process stopped.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Step 1: Generating table from papers...
python generate_table.py ./papers ./output
if %ERRORLEVEL% NEQ 0 (
    echo Error in generating table. Process stopped.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Step 2: Generating literature review...
python generate_literature_review.py --input "D:\RelatedWorkAutomated\output\meta_learning_related_work.csv" --output "D:\RelatedWorkAutomated\output\literature_review.md"
if %ERRORLEVEL% NEQ 0 (
    echo Error in generating literature review. Process stopped.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Process completed successfully!
echo Literature review has been generated at: D:\RelatedWorkAutomated\output\literature_review.md
pause 