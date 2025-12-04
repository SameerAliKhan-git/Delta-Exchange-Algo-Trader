@echo off
echo Starting Aladdin Command Center...

:: Start Backend
start "Aladdin Backend" cmd /k "cd dashboard\server && python server.py"

:: Start Frontend
start "Aladdin Frontend" cmd /k "cd dashboard && npm run dev"

echo.
echo 🚀 Dashboard launching...
echo 🌍 UI: http://localhost:5173
echo 🔌 API: http://localhost:8000
echo.
pause
