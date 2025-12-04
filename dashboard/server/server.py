import sys
import os
import threading
import asyncio
import logging
from typing import List, Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import TradingBot
try:
    from main import TradingBot
except ImportError as e:
    print(f"Error importing TradingBot: {e}")
    TradingBot = None

app = FastAPI(title="Aladdin Command Center API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
bot_instance = None
bot_thread = None
logs_buffer = []

# Logging Handler to capture logs
class ListHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        logs_buffer.append({
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": log_entry
        })
        if len(logs_buffer) > 1000:
            logs_buffer.pop(0)

# Setup logging capture
root_logger = logging.getLogger()
list_handler = ListHandler()
list_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(list_handler)

class BotStatus(BaseModel):
    status: str
    uptime: str
    pnl: float
    active_positions: int

@app.get("/api/status")
async def get_status():
    global bot_instance
    if not bot_instance:
        return {
            "status": "STOPPED",
            "uptime": "0s",
            "pnl": 0.0,
            "active_positions": 0
        }
    
    # Mock data for now if bot doesn't expose metrics directly yet
    return {
        "status": "RUNNING" if bot_instance._running else "STOPPED",
        "uptime": "Running", # Calculate actual uptime
        "pnl": 0.0, # Fetch from bot.metrics
        "active_positions": 0 # Fetch from bot.order_manager
    }

@app.post("/api/start")
async def start_bot():
    global bot_instance, bot_thread
    
    if bot_instance and bot_instance._running:
        return {"message": "Bot is already running"}
    
    if not TradingBot:
        raise HTTPException(status_code=500, detail="TradingBot class not found")
    
    try:
        bot_instance = TradingBot()
        
        def run_bot():
            try:
                bot_instance.start()
            except Exception as e:
                logging.error(f"Bot crashed: {e}")

        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        return {"message": "Bot started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop")
async def stop_bot():
    global bot_instance
    if bot_instance and bot_instance._running:
        bot_instance.stop()
        return {"message": "Bot stopped"}
    return {"message": "Bot is not running"}

@app.get("/api/logs")
async def get_logs(limit: int = 100):
    return logs_buffer[-limit:]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
