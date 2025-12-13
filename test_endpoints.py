#!/usr/bin/env python3
"""
Quick test script for /price and /forecast endpoints.
"""
import httpx
import json
import asyncio

async def test_endpoints():
    async with httpx.AsyncClient() as client:
        # Test /price endpoint
        print("\n=== Testing /price endpoint ===")
        try:
            response = await client.get("http://127.0.0.1:5000/price?product_id=1")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test /forecast endpoint
        print("\n=== Testing /forecast endpoint ===")
        try:
            response = await client.get("http://127.0.0.1:5000/forecast?product_id=1&months=3")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Product ID: {data['productId']}")
                print(f"Forecasting Window: {data['forecastingWindow']}")
                print(f"Total Revenue: {data['predictedTotalRevenue']}")
                print(f"Daily Predictions: {len(data['dailyPredictions'])} items")
            else:
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_endpoints())
