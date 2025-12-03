import requests, time, hmac, hashlib, json

API_KEY='vu55c1iSzUMwQSZwmPEMgHUVfJGpXo'
API_SECRET='gXzg0GMwqLTOhyzuSKf9G4OlOF1sX8RSJrcuPy0A98YofCWGcKEh8DoForAF'
BASE='https://api.india.delta.exchange'

def sign(path):
    ts=str(int(time.time()))
    sig=hmac.new(API_SECRET.encode(),('GET'+ts+path).encode(),hashlib.sha256).hexdigest()
    return {'api-key':API_KEY,'timestamp':ts,'signature':sig}

# Check positions
r=requests.get(BASE+'/v2/positions', headers=sign('/v2/positions'))
print('=== OPEN POSITIONS ===')
has_pos = False
for p in r.json().get('result',[]):
    size = p.get('size')
    if size and float(size) != 0:
        has_pos = True
        print(f"Symbol: {p.get('product_symbol')}")
        print(f"  Size: {size} contracts")
        print(f"  Side: {'LONG' if float(size) > 0 else 'SHORT'}")
        print(f"  Entry: ${p.get('entry_price')}")
        print(f"  Mark: ${p.get('mark_price')}")
        print(f"  UPNL: ${p.get('unrealized_pnl')}")
        print(f"  Margin: ${p.get('margin')}")
        print()

if not has_pos:
    print("No open positions")

# Check wallet
print('\n=== WALLET ===')
r=requests.get(BASE+'/v2/wallet/balances', headers=sign('/v2/wallet/balances'))
for b in r.json().get('result',[]):
    if float(b.get('balance',0)) > 0:
        print(f"Balance: ${b.get('balance')}")
        print(f"Available: ${b.get('available_balance')}")

# Check open orders
print('\n=== OPEN ORDERS ===')
r=requests.get(BASE+'/v2/orders', headers=sign('/v2/orders'))
orders = [o for o in r.json().get('result',[]) if o.get('state') == 'open']
if orders:
    for o in orders:
        print(f"{o.get('product_symbol')}: {o.get('side')} {o.get('size')} @ {o.get('limit_price')}")
else:
    print("No open orders")
