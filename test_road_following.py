#!/usr/bin/env python3
"""
Test script to verify road-following polyline generation
"""
import requests
import json
import polyline
import math
from geopy.distance import geodesic

def test_road_following():
    """Test the road-following polyline generation"""
    
    print("🚀 Testing Road-Following Polyline Generation")
    print("=" * 50)
    
    # Test data with multiple points
    test_data = """cluster,student_count,bus_stop_lat,bus_stop_lon
1,10,13.0827,80.2707
1,15,13.0828,80.2708
1,12,13.0829,80.2709
1,18,13.0830,80.2710"""
    
    # Create a file-like object
    from io import BytesIO
    file_obj = BytesIO(test_data.encode('utf-8'))
    file_obj.name = 'test_road.csv'
    
    # Test the API
    url = 'http://localhost:5000/api/optimize'
    files = {'file': ('test_road.csv', file_obj, 'text/csv')}
    
    try:
        print("Testing road-following polyline generation...")
        response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API call successful!")
            
            if data.get('success') and data.get('data', {}).get('routes'):
                routes = data['data']['routes']
                print(f"✅ Generated {len(routes)} routes")
                
                for i, route in enumerate(routes):
                    print(f"\n🚌 Route {i+1}:")
                    print(f"  - Bus ID: {route.get('bus_id')}")
                    print(f"  - Students: {route.get('total_students')}")
                    print(f"  - Distance: {route.get('distance_km')} km")
                    print(f"  - Duration: {route.get('duration_hours')} hours")
                    print(f"  - Has polyline: {bool(route.get('polyline'))}")
                    
                    if route.get('polyline'):
                        # Decode and analyze polyline
                        try:
                            decoded_points = polyline.decode(route['polyline'])
                            print(f"  - Polyline points: {len(decoded_points)}")
                            print(f"  - Polyline length: {len(route['polyline'])} characters")
                            
                            # Check if polyline has curves (more than direct line)
                            if len(decoded_points) > 2:
                                print(f"  - ✅ Road-following polyline detected!")
                                
                                # Calculate total distance of polyline
                                polyline_distance = 0
                                for j in range(len(decoded_points) - 1):
                                    dist = geodesic(decoded_points[j], decoded_points[j+1]).kilometers
                                    polyline_distance += dist
                                
                                print(f"  - Polyline distance: {polyline_distance:.2f} km")
                                print(f"  - Direct distance: {route.get('distance_km', 0):.2f} km")
                                
                                if polyline_distance > route.get('distance_km', 0):
                                    print(f"  - ✅ Polyline follows roads (longer than direct distance)")
                                else:
                                    print(f"  - ⚠️ Polyline may be direct line")
                            else:
                                print(f"  - ⚠️ Polyline has too few points")
                                
                        except Exception as e:
                            print(f"  - ❌ Error decoding polyline: {e}")
                    else:
                        print(f"  - ❌ No polyline generated")
                
                # Check overall results
                routes_with_polylines = sum(1 for route in routes if route.get('polyline'))
                print(f"\n📊 Summary:")
                print(f"  - Routes with polylines: {routes_with_polylines}/{len(routes)}")
                
                if routes_with_polylines == len(routes):
                    print("  - ✅ All routes have road-following polylines!")
                else:
                    print("  - ⚠️ Some routes missing polylines")
                    
            else:
                print("❌ No routes generated")
                print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_polyline_quality():
    """Test the quality of generated polylines"""
    
    print("\n🔍 Testing Polyline Quality")
    print("=" * 30)
    
    # Test coordinates around Chennai
    test_coords = [
        (13.0827, 80.2707),  # College
        (13.0828, 80.2708),  # Point 1
        (13.0829, 80.2709),  # Point 2
        (13.0830, 80.2710),  # Point 3
        (13.0827, 80.2707)   # Back to college
    ]
    
    # Create polyline with curves
    polyline_points = []
    
    for i in range(len(test_coords) - 1):
        start_coord = test_coords[i]
        end_coord = test_coords[i + 1]
        
        # Add start point
        polyline_points.append(start_coord)
        
        # Calculate distance
        distance = geodesic(start_coord, end_coord).kilometers
        
        # Add intermediate points with curves
        if distance > 0.5:
            num_intermediate = min(int(distance * 3), 10)
            
            for j in range(1, num_intermediate + 1):
                fraction = j / (num_intermediate + 1)
                
                # Interpolate
                lat = start_coord[0] + (end_coord[0] - start_coord[0]) * fraction
                lon = start_coord[1] + (end_coord[1] - start_coord[1]) * fraction
                
                # Add curve
                if distance > 1:
                    curve_intensity = min(distance * 0.001, 0.003)
                    curve_offset_lat = curve_intensity * math.sin(fraction * math.pi * 2)
                    curve_offset_lon = curve_intensity * math.cos(fraction * math.pi * 2)
                    lat += curve_offset_lat
                    lon += curve_offset_lon
                
                polyline_points.append((lat, lon))
        
        # Add end point
        polyline_points.append(end_coord)
    
    # Encode polyline
    encoded_polyline = polyline.encode(polyline_points)
    
    print(f"✅ Generated polyline with {len(polyline_points)} points")
    print(f"✅ Encoded polyline length: {len(encoded_polyline)} characters")
    
    # Decode and verify
    decoded_points = polyline.decode(encoded_polyline)
    print(f"✅ Decoded polyline has {len(decoded_points)} points")
    
    # Calculate total distance
    total_distance = 0
    for i in range(len(decoded_points) - 1):
        dist = geodesic(decoded_points[i], decoded_points[i+1]).kilometers
        total_distance += dist
    
    print(f"✅ Total polyline distance: {total_distance:.2f} km")
    
    # Calculate direct distance
    direct_distance = geodesic(test_coords[0], test_coords[-1]).kilometers
    print(f"✅ Direct distance: {direct_distance:.2f} km")
    
    if total_distance > direct_distance:
        print("✅ Polyline follows roads (longer than direct distance)")
    else:
        print("⚠️ Polyline may be too direct")

if __name__ == "__main__":
    test_polyline_quality()
    print()
    test_road_following() 