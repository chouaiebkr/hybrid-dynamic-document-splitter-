import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:5000"

def test_api():
    """Test the API endpoints"""
    
    print("Testing CAR-Based Semantic Search API...")
    
    # 1. Test status
    print("\n1. Testing status endpoint...")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 2. Setup database (if not already done)
    print("\n2. Setting up database...")
    response = requests.post(f"{BASE_URL}/setup")
    print(f"Setup: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 3. Process documents
    print("\n3. Processing documents...")
    response = requests.post(f"{BASE_URL}/documents/process")
    print(f"Process: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 4. Check status again
    print("\n4. Checking status after processing...")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # 5. Perform some searches
    print("\n5. Testing search functionality...")
    
    test_queries = [
        "financial data",
        "project status",
        "team members",
        "revenue summary"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        response = requests.get(f"{BASE_URL}/search?q={query}&top_k=3")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {results['count']} results:")
            for i, result in enumerate(results['results'], 1):
                print(f"  {i}. [Similarity: {result['similarity']:.4f}] - {result['source']}")
                print(f"      {result['text'][:100]}...")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        time.sleep(1)  # Brief pause between queries

if __name__ == "__main__":
    test_api()