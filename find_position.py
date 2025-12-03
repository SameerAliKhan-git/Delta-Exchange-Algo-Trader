import requests, time, hmac, hashlib, json

API_KEY='vu55c1iSzUMwQSZwmPEMgHUVfJGpXo'
API_SECRET='gXzg0GMwqLTOhyzuSKf9G4OlOF1sX8RSJrcuPy0A98YofCWGcKEh8DoForAF'
BASE='https://api.india.delta.exchange'

def sign(method, path, payload=""):
    ts=str(int(time.time()))
    sig=hmac.new(API_SECRET.encode(),(method+ts+path+payload).encode(),hashlib.sha256).hexdigest()
    return {'api-key':API_KEY,'timestamp':ts,'signature':sig,'Content-Type':'application/json'}

# Get positions with margining parameter
print("=== POSITIONS (with margining) ===")
r=requests.get(BASE+'/v2/positions/margined', headers=sign('GET','/v2/positions/margined'))
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

# Try different endpoint
print("\n=== POSITIONS (standard) ===")
r=requests.get(BASE+'/v2/positions', headers=sign('GET','/v2/positions'))
data = r.json()
print(f"Success: {data.get('success')}")
if data.get('result'):
    for p in data['result']:
        print(f"Product: {p.get('product_id')} {p.get('product_symbol')}")
        print(f"  Size: {p.get('size')}")
        print(f"  Entry: {p.get('entry_price')}")
        print(f"  Margin: {p.get('margin')}")
        print()
else:
    print("No positions or error")
    print(json.dumps(data, indent=2))

# Get product list to see what products exist
print("\n=== CHECKING SPECIFIC PRODUCTS ===")
products = [14969, 3136, 27, 14823]  # XRPUSD, ETHUSD, BTCUSD, SOLUSD
for pid in products:
    r=requests.get(BASE+f'/v2/positions?product_id={pid}', headers=sign('GET',f'/v2/positions?product_id={pid}'))
    data = r.json()
    if data.get('result'):
        for p in data['result']:
            size = p.get('size', 0)
            if float(size) != 0:
                print(f"FOUND: Product {pid}: size={size}, margin={p.get('margin')}")
