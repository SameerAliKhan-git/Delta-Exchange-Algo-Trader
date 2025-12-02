"""Test Delta Exchange API Connection"""
from config.credentials import API_KEY, API_SECRET, BASE_URL
import requests
import hmac
import hashlib
import time
import json

print('='*60)
print('DELTA EXCHANGE ACCOUNT TEST')
print('='*60)
print(f'API Key: {API_KEY[:10]}...{API_KEY[-5:]}')
print(f'Base URL: {BASE_URL}')

# Generate signature
timestamp = str(int(time.time()))
method = 'GET'
endpoint = '/v2/wallet/balances'
signature_data = method + timestamp + endpoint
signature = hmac.new(
    API_SECRET.encode('utf-8'),
    signature_data.encode('utf-8'),
    hashlib.sha256
).hexdigest()

headers = {
    'api-key': API_KEY,
    'signature': signature,
    'timestamp': timestamp,
    'Content-Type': 'application/json'
}

print('\nFetching wallet balances...')
response = requests.get(f'{BASE_URL}{endpoint}', headers=headers, timeout=30)
print(f'Status: {response.status_code}')

if response.status_code == 200:
    data = response.json()
    if 'result' in data:
        print('\n💰 WALLET BALANCES:')
        total_usd = 0
        for asset in data['result']:
            balance = float(asset.get('balance', 0))
            available = float(asset.get('available_balance', 0))
            symbol = asset.get('asset_symbol', 'Unknown')
            if balance > 0 or available > 0:
                print(f'  {symbol}: Balance={balance}, Available={available}')
                if symbol in ['USDT', 'USD', 'USDC']:
                    total_usd += available
        print(f'\n📊 Total Available (USD): ${total_usd:,.2f}')
    else:
        print(f'Response: {data}')
else:
    print(f'Error: {response.text}')

# Get positions (need to specify product_id or underlying_asset_symbol)
print('\nFetching positions...')
endpoint = '/v2/positions?underlying_asset_symbol=BTC'
timestamp = str(int(time.time()))
signature_data = method + timestamp + endpoint
signature = hmac.new(
    API_SECRET.encode('utf-8'),
    signature_data.encode('utf-8'),
    hashlib.sha256
).hexdigest()
headers['signature'] = signature
headers['timestamp'] = timestamp

response = requests.get(f'{BASE_URL}{endpoint}', headers=headers, timeout=30)
if response.status_code == 200:
    data = response.json()
    positions = [p for p in data.get('result', []) if float(p.get('size', 0)) != 0]
    print(f'📊 Open BTC Positions: {len(positions)}')
    for p in positions:
        print(f"  {p.get('symbol')}: Size={p.get('size')}, Entry={p.get('entry_price')}")
else:
    print(f'Positions Error: {response.text}')

# Get current BTC price
print('\nFetching BTC price...')
response = requests.get(f'{BASE_URL}/v2/tickers/BTCUSD', timeout=10)
if response.status_code == 200:
    ticker = response.json().get('result', {})
    price = ticker.get('mark_price', ticker.get('close', 'N/A'))
    print(f'📈 BTCUSD Price: ${float(price):,.2f}' if price != 'N/A' else 'Price unavailable')

# Get ETH price  
response = requests.get(f'{BASE_URL}/v2/tickers/ETHUSD', timeout=10)
if response.status_code == 200:
    ticker = response.json().get('result', {})
    price = ticker.get('mark_price', ticker.get('close', 'N/A'))
    print(f'📈 ETHUSD Price: ${float(price):,.2f}' if price != 'N/A' else 'Price unavailable')

print('\n' + '='*60)
print('✅ API Connection Test Complete')
print('='*60)
