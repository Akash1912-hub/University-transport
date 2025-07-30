#!/usr/bin/env python3
"""
Test script to verify performance improvements for large datasets
"""

import requests
import time
import json
import pandas as pd
from io import StringIO

def create_test_data(num_clusters=25):
    """Create test CSV data with multiple clusters"""
    data = []
    
    # College location
    college_lat, college_lon = 13.0827, 80.2707
    
    # Generate clusters around Chennai
    for cluster_id in range(1, num_clusters + 1):
        # Create 3-8 points per cluster
        num_points = 3 + (cluster_id % 6)
        
        for point_id in range(num_points):
            # Spread points around Chennai
            lat_offset = (cluster_id - 1) * 0.02  # 2km spacing
            lon_offset = (point_id - 1) * 0.01    # 1km spacing
            
            lat = college_lat + lat_offset + (point_id * 0.005)
            lon = college_lon + lon_offset + (point_id * 0.003)
            
            # Ensure coordinates are within reasonable bounds
            lat = max(12.5, min(13.5, lat))
            lon = max(79.5, min(81.0, lon))
            
            data.append({
                'cluster': f'cluster_{cluster_id}',
                'student_count': 10 + (point_id * 5),  # 10-35 students per point
                'bus_stop_lat': round(lat, 6),
                'bus_stop_lon': round(lon, 6)
            })
    
    return pd.DataFrame(data)

def test_performance():
    """Test the performance improvements"""
    print("🚀 Testing Performance Improvements")
    print("=" * 50)
    
    # Test with different cluster sizes
    test_sizes = [10, 15, 20, 25]
    
    for num_clusters in test_sizes:
        print(f"\n📊 Testing with {num_clusters} clusters...")
        
        # Create test data
        df = create_test_data(num_clusters)
        
        # Convert to CSV string
        csv_content = df.to_csv(index=False)
        
        # Test API
        start_time = time.time()
        
        try:
            response = requests.post(
                'http://localhost:5000/api/optimize',
                files={'file': ('test_data.csv', csv_content, 'text/csv')},
                timeout=300  # 5 minutes timeout
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    summary = data.get('data', {})
                    routes_generated = summary.get('summary', {}).get('total_buses_needed', 0)
                    api_processing_time = summary.get('processing_time_seconds', 0)
                    clusters_per_minute = summary.get('performance', {}).get('clusters_per_minute', 0)
                    
                    print(f"✅ Success: {routes_generated} routes generated")
                    print(f"⏱️  API Processing Time: {api_processing_time:.1f}s")
                    print(f"⏱️  Total Time (including network): {processing_time:.1f}s")
                    print(f"⚡ Speed: {clusters_per_minute:.1f} clusters/minute")
                    
                    # Performance analysis
                    if processing_time < 60:
                        print(f"🎉 Excellent! {num_clusters} clusters processed in under 1 minute")
                    elif processing_time < 120:
                        print(f"👍 Good! {num_clusters} clusters processed in under 2 minutes")
                    else:
                        print(f"⚠️  Slow: {num_clusters} clusters took {processing_time/60:.1f} minutes")
                        
                else:
                    print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ Timeout after 5 minutes for {num_clusters} clusters")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Performance test completed!")

if __name__ == "__main__":
    test_performance() 