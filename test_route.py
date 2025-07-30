#!/usr/bin/env python3
"""
Test script to verify route optimization fixes
"""
import requests
import json

def test_route_optimization():
    """Test the route optimization API"""
    
    # Test data - single point route
    test_data = """cluster,student_count,bus_stop_lat,bus_stop_lon
1,5,13.0828,80.2708"""
    
    # Create a file-like object
    from io import BytesIO
    file_obj = BytesIO(test_data.encode('utf-8'))
    file_obj.name = 'test.csv'
    
    # Test the API
    url = 'http://localhost:5000/api/optimize'
    files = {'file': ('test.csv', file_obj, 'text/csv')}
    
    try:
        print("Testing route optimization API...")
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API call successful!")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            if data.get('success') and data.get('data', {}).get('routes'):
                routes = data['data']['routes']
                print(f"✅ Generated {len(routes)} routes")
                
                for i, route in enumerate(routes):
                    print(f"Route {i+1}:")
                    print(f"  - Bus ID: {route.get('bus_id')}")
                    print(f"  - Students: {route.get('total_students')}")
                    print(f"  - Distance: {route.get('distance_km')} km")
                    print(f"  - Duration: {route.get('duration_hours')} hours")
                    print(f"  - Stops: {len(route.get('stops', []))}")
                    print(f"  - Has polyline: {bool(route.get('polyline'))}")
                    if route.get('polyline'):
                        print(f"  - Polyline length: {len(route['polyline'])}")
            else:
                print("❌ No routes generated")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_osrm_connectivity():
    """Test OSRM connectivity"""
    try:
        print("Testing OSRM connectivity...")
        response = requests.get('http://localhost:5000/api/test-osrm')
        
        if response.status_code == 200:
            data = response.json()
            print(f"OSRM test result: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ OSRM test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing OSRM: {e}")

if __name__ == "__main__":
    print("🚀 Testing Route Optimization Fixes")
    print("=" * 50)
    
    test_osrm_connectivity()
    print()
    test_route_optimization() 