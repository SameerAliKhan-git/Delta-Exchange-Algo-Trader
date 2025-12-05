"""
Delta Exchange API Credentials
================================
SECURITY WARNING: Keep this file private. Never commit to public repositories.
Add to .gitignore!
"""

# Delta Exchange India API Credentials
# Full Trading + Data Access API
API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"

# Set to False for LIVE trading, True for testnet/demo
TESTNET = False

# API Endpoints
MAINNET_REST = "https://api.india.delta.exchange"
MAINNET_WS = "wss://socket.india.delta.exchange"
TESTNET_REST = "https://cdn-ind.testnet.deltaex.org"  
TESTNET_WS = "wss://socket-ind.testnet.deltaex.org"

# Default BASE_URL for convenience
BASE_URL = MAINNET_REST if not TESTNET else TESTNET_REST

def get_rest_url():
    return TESTNET_REST if TESTNET else MAINNET_REST

def get_ws_url():
    return TESTNET_WS if TESTNET else MAINNET_WS
