import requests, time, hmac, hashlib, json

API_KEY='vu55c1iSzUMwQSZwmPEMgHUVfJGpXo'
API_SECRET='gXzg0GMwqLTOhyzuSKf9G4OlOF1sX8RSJrcuPy0A98YofCWGcKEh8DoForAF'
BASE='https://api.india.delta.exchange'

def sign(method, path, payload=""):
    ts=str(int(time.time()))
    sig=hmac.new(API_SECRET.encode(),(method+ts+path+payload).encode(),hashlib.sha256).hexdigest()
    return {'api-key':API_KEY,'timestamp':ts,'signature':sig,'Content-Type':'application/json'}

# Check ALL positions with full details
print("=== CHECKING ALL POSITIONS ===")
r=requests.get(BASE+'/v2/positions', headers=sign('GET','/v2/positions'))
data = r.json()
print(f"API Success: {data.get('success')}")
for p in data.get('result',[]):
    sym = p.get('product_symbol')
    size = p.get('size')
    margin = p.get('margin', 0)
    print(f"{sym}: size={size}, margin=${margin}")

# Check orders
print("\n=== CHECKING ALL ORDERS ===")
r=requests.get(BASE+'/v2/orders', headers=sign('GET','/v2/orders'))
for o in r.json().get('result',[]):
    print(f"{o.get('product_symbol')}: {o.get('state')} - {o.get('side')} {o.get('size')} @ {o.get('limit_price')}")

# Check margin breakdown
print("\n=== WALLET DETAILS ===")
r=requests.get(BASE+'/v2/wallet/balances', headers=sign('GET','/v2/wallet/balances'))
for b in r.json().get('result',[]):
    if b.get('asset_symbol') == 'USD':
        print(f"Balance: ${b.get('balance')}")
        print(f"Available: ${b.get('available_balance')}")
        print(f"Order Margin: ${b.get('order_margin', 0)}")
        print(f"Position Margin: ${b.get('position_margin', 0)}")
        print(f"Commission: ${b.get('commission', 0)}")
        print(f"All fields: {json.dumps(b, indent=2)}")

# Cancel all open orders to free up margin
print("\n=== CANCELLING ALL OPEN ORDERS ===")
payload = json.dumps({"cancel_all": True})
r=requests.delete(BASE+'/v2/orders/all', headers=sign('DELETE','/v2/orders/all', payload), data=payload)
print(f"Cancel result: {r.json()}")

# Check balance again
print("\n=== BALANCE AFTER CANCEL ===")
time.sleep(1)
r=requests.get(BASE+'/v2/wallet/balances', headers=sign('GET','/v2/wallet/balances'))
for b in r.json().get('result',[]):
    if b.get('asset_symbol') == 'USD':
        print(f"Balance: ${b.get('balance')}")
        print(f"Available: ${b.get('available_balance')}")
