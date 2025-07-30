#!/usr/bin/env python3
"""
Test script for large dataset (400+ boarding points) to verify algorithm performance
"""
import requests
import json
import pandas as pd
import numpy as np
import random
from io import StringIO

def generate_large_test_data(num_points=400):
    """Generate test data with 400+ boarding points around Chennai"""
    
    # College coordinates (Rajalakshmi Engineering College)
    college_lat, college_lon = 13.0827, 80.2707
    
    # Generate points in a radius around Chennai
    points = []
    
    for i in range(num_points):
        # Generate random coordinates within ~50km radius of college
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(2, 50)  # 2-50 km from college
        
        # Convert to lat/lon offset
        lat_offset = distance * np.cos(angle) / 111  # ~111 km per degree latitude
        lon_offset = distance * np.sin(angle) / (111 * np.cos(np.radians(college_lat)))
        
        lat = college_lat + lat_offset
        lon = college_lon + lon_offset
        
        # Ensure coordinates are within reasonable bounds
        lat = max(12.5, min(13.5, lat))
        lon = max(79.5, min(81.0, lon))
        
        # Generate realistic student counts (5-25 students per stop)
        student_count = random.randint(5, 25)
        
        points.append({
            'cluster': f'cluster_{i//10 + 1}',  # Group into clusters
            'student_count': student_count,
            'bus_stop_lat': round(lat, 6),
            'bus_stop_lon': round(lon, 6)
        })
    
    return points

def test_large_dataset():
    """Test the algorithm with 400+ boarding points"""
    
    print("🚀 Testing Large Dataset Algorithm")
    print("=" * 50)
    
    # Generate test data
    print("📊 Generating test data...")
    test_points = generate_large_test_data(400)
    
    # Create CSV content
    csv_content = "cluster,student_count,bus_stop_lat,bus_stop_lon\n"
    for point in test_points:
        csv_content += f"{point['cluster']},{point['student_count']},{point['bus_stop_lat']},{point['bus_stop_lon']}\n"
    
    print(f"✅ Generated {len(test_points)} boarding points")
    print(f"📈 Total students: {sum(p['student_count'] for p in test_points)}")
    
    # Test the API
    print("\n🔄 Testing API with large dataset...")
    
    # Create a file-like object
    from io import BytesIO
    file_obj = BytesIO(csv_content.encode('utf-8'))
    file_obj.name = 'large_test.csv'
    
    # Test the API
    url = 'http://localhost:5000/api/optimize'
    files = {'file': ('large_test.csv', file_obj, 'text/csv')}
    
    try:
        response = requests.post(url, files=files, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API call successful!")
            
            if data.get('success') and data.get('data', {}).get('routes'):
                routes = data['data']['routes']
                summary = data['data']['summary']
                
                print(f"\n📊 Results Summary:")
                print(f"  - Total buses needed: {summary['total_buses_needed']}")
                print(f"  - Total students covered: {summary['total_students_covered']}")
                print(f"  - Total fleet distance: {summary['total_fleet_distance_km']} km")
                print(f"  - Total fleet duration: {summary['total_fleet_duration_hours']} hours")
                print(f"  - Average students per bus: {summary['average_students_per_bus']}")
                print(f"  - Capacity utilization: {summary['capacity_utilization']}%")
                print(f"  - Processing time: {data['data']['processing_time_seconds']} seconds")
                
                print(f"\n🚌 Route Details:")
                for i, route in enumerate(routes[:5]):  # Show first 5 routes
                    print(f"  Bus {route['bus_id']}: {route['total_students']} students, "
                          f"{route['total_stops']} stops, {route['distance_km']} km, "
                          f"{route['duration_hours']} hours")
                
                if len(routes) > 5:
                    print(f"  ... and {len(routes) - 5} more routes")
                
                # Verify capacity constraints
                capacity_violations = 0
                for route in routes:
                    if route['total_students'] > 55:
                        capacity_violations += 1
                        print(f"⚠️ Bus {route['bus_id']} exceeds capacity: {route['total_students']} students")
                
                if capacity_violations == 0:
                    print("✅ All routes respect capacity constraints")
                else:
                    print(f"❌ {capacity_violations} routes exceed capacity")
                
                # Check polyline generation
                routes_with_polylines = sum(1 for route in routes if route.get('polyline'))
                print(f"✅ {routes_with_polylines}/{len(routes)} routes have road-following polylines")
                
            else:
                print("❌ No routes generated")
                print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_google_api_connectivity():
    """Test Google API connectivity"""
    try:
        print("🔍 Testing Google API connectivity...")
        response = requests.get('http://localhost:5000/api/health')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['message']}")
            print(f"📋 Features: {', '.join(data['features'])}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing connectivity: {e}")

if __name__ == "__main__":
    test_google_api_connectivity()
    print()
    test_large_dataset() 