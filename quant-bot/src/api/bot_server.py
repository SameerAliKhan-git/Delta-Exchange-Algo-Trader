"""
Bot API Server
==============
Exposes REST API for the Autonomous Trading Bot.
Handles:
- Signal ingestion (News, Social, etc.)
- Bot control (Start/Stop)
- Status monitoring
- Metrics export
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.engine.autonomous_orchestrator import AutonomousOrchestrator, OrchestratorConfig, SystemState

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BotAPI")

# Global Bot Instance
bot_orchestrator: Optional[AutonomousOrchestrator] = None
bot_task: Optional[asyncio.Task] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global bot_orchestrator, bot_task
    logger.info("Starting Bot API Server...")
    
    config = OrchestratorConfig()
    bot_orchestrator = AutonomousOrchestrator(config)
    
    # Start bot loop in background
    bot_task = asyncio.create_task(bot_orchestrator.run())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Bot API Server...")
    if bot_orchestrator:
        bot_orchestrator.state = SystemState.SHUTDOWN
    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="Delta Algo Bot API", lifespan=lifespan)

# Data Models
class NewsSignal(BaseModel):
    timestamp: str
    source: str
    url: str
    title: str
    event: str
    tickers: list[str]
    sentiment: Dict[str, Any]
    score: float
    confidence: float
    decay_seconds: int
    impact: float
    raw_article: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "ok", "bot_state": bot_orchestrator.state.value if bot_orchestrator else "unknown"}

@app.post("/api/signals/news")
async def receive_news_signal(signal: NewsSignal):
    """
    Receive news signals from the sentiment pipeline.
    """
    if not bot_orchestrator:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    logger.info(f"Received news signal: {signal.title} (Score: {signal.score})")
    
    # Map tickers to symbols (simple mapping for now)
    # In production, use ProductDiscovery to map 'BTC' -> 'BTCUSDT'
    symbols = []
    for ticker in signal.tickers:
        if ticker in ["BTC", "BITCOIN"]:
            symbols.append("BTCUSDT")
        elif ticker in ["ETH", "ETHEREUM"]:
            symbols.append("ETHUSDT")
        elif ticker in ["SOL", "SOLANA"]:
            symbols.append("SOLUSDT")
    
    # Add signal to aggregator for each mapped symbol
    for symbol in symbols:
        bot_orchestrator.signal_aggregator.add_signal(
            source="news_sentiment",
            symbol=symbol,
            signal=signal.score,
            confidence=signal.confidence,
            features={
                "event": signal.event,
                "impact": signal.impact,
                "source": signal.source
            }
        )
        
    # Handle Emergency Hedges (Direct Action)
    if signal.event in ["exchange_hack", "exploit", "bankruptcy"] and signal.score < -0.7:
        logger.warning(f"CRITICAL EVENT: {signal.event} - Triggering safety checks")
        # Here we would call risk_controller.trigger_emergency_protocol()
        # For now, we just log it.
        
    return {"status": "accepted", "mapped_symbols": symbols}

@app.get("/api/status")
async def get_bot_status():
    if not bot_orchestrator:
        return {"status": "offline"}
    
    return {
        "state": bot_orchestrator.state.value,
        "positions": len(bot_orchestrator.risk_controller.positions),
        "daily_pnl": bot_orchestrator.risk_controller.daily_pnl,
        "errors": bot_orchestrator.errors
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
