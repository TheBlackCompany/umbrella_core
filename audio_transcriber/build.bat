@echo off
REM Build Umbrella Transcriber Docker image for Windows

echo Building Umbrella Transcriber Docker image...
echo ========================================

REM Check if .env exists
if not exist .env (
    echo ERROR: .env file not found!
    echo Please copy .env.template to .env and configure it
    exit /b 1
)

REM Build the image
docker build -t umbrella-transcriber:latest .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
    echo To run: docker-compose -f docker-compose.prod.yml up -d
) else (
    echo.
    echo Build failed!
    exit /b 1
)