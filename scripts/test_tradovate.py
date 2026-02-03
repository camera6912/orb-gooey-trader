#!/usr/bin/env python3
"""Test Tradovate API connection and order placement.

Usage:
    python scripts/test_tradovate.py           # Test connection only
    python scripts/test_tradovate.py --order   # Place a test order (DEMO only!)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from broker.tradovate import load_tradovate_client


def main():
    parser = argparse.ArgumentParser(description="Test Tradovate API")
    parser.add_argument("--order", action="store_true", help="Place a test order (DEMO only!)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Tradovate API Connection Test")
    print("=" * 50)
    
    # Load client from secrets
    client = load_tradovate_client("config/secrets.yaml")
    
    if not client:
        print("❌ Failed to load Tradovate client")
        print("   Make sure config/secrets.yaml has tradovate section")
        return 1
    
    print(f"✓ Client loaded for {client.environment.upper()} environment")
    print(f"  Base URL: {client.base_url}")
    
    # Test authentication
    print("\n--- Testing Authentication ---")
    if client.authenticate():
        print(f"✅ Authentication successful!")
        print(f"   User ID: {client._token.user_id}")
        print(f"   Token expires: {client._token.expires_at}")
    else:
        print("❌ Authentication failed")
        return 1
    
    # Get accounts
    print("\n--- Getting Accounts ---")
    accounts = client.get_accounts()
    if accounts:
        for acc in accounts:
            print(f"✓ Account: {acc.get('name')} (ID: {acc.get('id')})")
            print(f"  Balance: ${acc.get('cashBalance', 0):,.2f}")
    else:
        print("⚠️ No accounts found")
    
    # Get contract info
    print("\n--- Testing Contract Lookup ---")
    for symbol in ["/NQ", "/ES", "/MNQ"]:
        contract_id = client.get_contract_id(symbol)
        if contract_id:
            print(f"✓ {symbol} -> Contract ID: {contract_id}")
        else:
            print(f"⚠️ {symbol} -> Not found")
    
    # Get positions
    print("\n--- Current Positions ---")
    positions = client.get_positions()
    if positions:
        for pos in positions:
            net = pos.get("netPos", 0)
            if net != 0:
                print(f"✓ Position: {net} contracts")
    else:
        print("  No open positions")
    
    # Test order (only if --order flag and DEMO mode)
    if args.order:
        print("\n--- Test Order Placement ---")
        
        if client.environment != "demo":
            print("❌ Test orders only allowed in DEMO mode!")
            return 1
        
        print("⚠️ Placing test order: Buy 1 /MNQ (Micro NQ) at Market")
        print("   This is a DEMO order - no real money involved")
        
        confirm = input("   Proceed? (y/n): ")
        if confirm.lower() != "y":
            print("   Cancelled")
            return 0
        
        result = client.place_order(
            symbol="/MNQ",
            action="Buy",
            quantity=1,
            order_type="Market",
        )
        
        if result.success:
            print(f"✅ Order placed! Order ID: {result.order_id}")
            
            # Close the position immediately
            print("   Closing position...")
            close_result = client.place_order(
                symbol="/MNQ",
                action="Sell",
                quantity=1,
                order_type="Market",
            )
            if close_result.success:
                print(f"✅ Position closed! Order ID: {close_result.order_id}")
            else:
                print(f"⚠️ Close failed: {close_result.error}")
        else:
            print(f"❌ Order failed: {result.error}")
    
    print("\n" + "=" * 50)
    print("✅ Tradovate API test complete!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
