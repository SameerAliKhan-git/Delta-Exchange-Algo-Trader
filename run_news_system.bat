@echo off
echo Starting Delta Algo News System...

REM Start the Bot API Server (which runs the bot)
start "Bot API Server" cmd /k "cd quant-bot && uvicorn src.api.bot_server:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for server to start
timeout /t 10

REM Start the News Pipeline
start "News Pipeline" cmd /k "cd quant-bot && python src/news/news_sentiment_pipeline.py"

echo System started.
echo Bot API: http://localhost:8000/docs
echo News Pipeline is running in background.
