"""
Aladdin Trading Bot - Quick Start Guide
========================================

⚠️ IP WHITELISTING REQUIRED
============================

Your Delta Exchange API key requires IP whitelisting.
Current IP: 14.139.82.36

To enable trading, follow these steps:

1. Log in to Delta Exchange India (https://india.delta.exchange)
2. Go to Settings → API Keys
3. Find your API key: TcYH2ep58n...
4. Click "Edit" or "Manage"
5. Add your IP address to the whitelist: 14.139.82.36
6. Save changes

Note: You can add multiple IPs or ranges if needed.

Alternative: Create a new API key WITHOUT IP restriction
(less secure but more flexible for testing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONCE IP IS WHITELISTED, RUN:

    python aladdin_autonomous.py

This will:
✅ Connect to your Delta Exchange account
✅ Read your account balance
✅ Monitor global crypto news
✅ Analyze market sentiment
✅ Execute trades automatically (long/short)
✅ Manage positions with stop-loss and take-profit
✅ Account for trading fees in profit calculations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADING CONFIGURATION (in aladdin_autonomous.py):

- Max position size: 10% of balance
- Stop loss: 2%
- Take profit: 4%
- Max daily loss: 5%
- Max positions: 3
- Leverage: 5x
- Trading symbols: BTCUSD, ETHUSD, SOLUSD

TRADING FEES (Delta Exchange India):
- Maker fee: 0.02%
- Taker fee: 0.05%

The bot calculates minimum profit targets AFTER fees to ensure
you don't lose money on fees alone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK WARNING:
=============
⚠️ Cryptocurrency trading involves significant risk
⚠️ Past performance is not indicative of future results
⚠️ Never trade with money you cannot afford to lose
⚠️ This bot is for educational purposes
⚠️ Start with small amounts to test

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(__doc__)

# Check current IP
import requests
try:
    ip_response = requests.get('https://api.ipify.org?format=json', timeout=5)
    current_ip = ip_response.json().get('ip', 'Unknown')
    print(f"🌐 Your Current IP: {current_ip}")
    print(f"\n📝 Add this IP to your Delta Exchange API whitelist!")
except:
    print("Could not determine current IP")
