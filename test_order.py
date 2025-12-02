"""
Test Order Placement on Delta Exchange India
"""
import hmac
import hashlib
import time
import json
import requests

# API Configuration
API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"

def generate_signature(method: str, endpoint: str, payload: str, timestamp: str) -> str:
    """Generate API signature"""
    message = f"{method}{timestamp}{endpoint}{payload}"
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return signature

def make_request(method: str, endpoint: str, data: dict = None):
    """Make authenticated API request"""
    timestamp = str(int(time.time()))
    payload = json.dumps(data) if data else ""
    signature = generate_signature(method, endpoint, payload, timestamp)
    
    headers = {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "Content-Type": "application/json"
    }
    
    url = BASE_URL + endpoint
    
    print(f"\n{'='*60}")
    print(f"Request: {method} {endpoint}")
    print(f"Payload: {payload}")
    print(f"Headers: api-key, timestamp, signature present")
    
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, headers=headers, data=payload)
    
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body: {response.text}")
    print(f"{'='*60}")
    
    return response

def get_product_id(symbol: str) -> int:
    """Get product ID for a symbol"""
    response = requests.get(f"{BASE_URL}/v2/products")
    if response.status_code == 200:
        products = response.json().get("result", [])
        for p in products:
            if p.get("symbol") == symbol:
                print(f"Product {symbol}: ID={p.get('id')}")
                print(f"  Min Size: {p.get('contract_value')}")
                print(f"  Contract Type: {p.get('contract_type')}")
                print(f"  Tick Size: {p.get('tick_size')}")
                return p.get("id")
    return None

def get_balance():
    """Get account balance"""
    response = make_request("GET", "/v2/wallet/balances")
    if response.status_code == 200:
        result = response.json()
        for asset in result.get("result", []):
            symbol = asset.get("asset_symbol")
            available = asset.get("available_balance")
            print(f"  {symbol}: {available}")
    return response

def test_order():
    """Test placing a tiny market order"""
    
    print("\n" + "="*60)
    print("DELTA EXCHANGE ORDER TEST")
    print("="*60)
    
    # Get balance first
    print("\n1. Checking balance...")
    get_balance()
    
    # Get product info
    print("\n2. Getting ETHUSD product info...")
    product_id = get_product_id("ETHUSD")
    
    if not product_id:
        print("Could not find ETHUSD product")
        return
    
    # Try placing a tiny order
    print("\n3. Attempting to place order...")
    
    # Delta Exchange uses contracts, not raw crypto amounts
    # For perpetuals, size is in USD notional / contract value
    order_data = {
        "product_id": product_id,
        "side": "buy",
        "size": 1,  # Minimum order size is usually 1 contract
        "order_type": "market_order"
    }
    
    response = make_request("POST", "/v2/orders", order_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ ORDER PLACED SUCCESSFULLY!")
        print(f"Order ID: {result.get('result', {}).get('id')}")
    else:
        print(f"\n❌ ORDER FAILED")
        try:
            error = response.json()
            print(f"Error: {error}")
        except:
            print(f"Raw: {response.text}")

if __name__ == "__main__":
    test_order()
