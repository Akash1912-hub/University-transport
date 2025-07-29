from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import json
import tempfile
import os
import math
from functools import lru_cache
import time

app = Flask(__name__)
CORS(app)

# === Constants ===
COLLEGE_LAT = 13.0827  # Rajalakshmi Engineering College, Chennai
COLLEGE_LON = 80.2707
MAX_BUS_CAPACITY = 55
MAX_DISTANCE_KM = 40
MAX_DURATION_MIN = 120
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

# Cache for distance calculations
distance_cache = {}

def calculate_optimal_buses(total_students):
    return max(1, min(20, (total_students + MAX_BUS_CAPACITY - 1) // MAX_BUS_CAPACITY))

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two points"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def batch_distance_matrix(coords, batch_size=10):
    """Get distance matrix using batch API calls for efficiency"""
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    
    print(f"Computing distance matrix for {n} points using batch processing...")
    
    # Use straight-line distance for small datasets (faster)
    if n <= 20:
        print("Using straight-line distance calculation for small dataset")
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = haversine_distance(
                        coords[i][0], coords[i][1], 
                        coords[j][0], coords[j][1]
                    )
        return matrix
    
    # For larger datasets, use batch API calls
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        print(f"Processing batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size}")
        
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            
            # Build batch URL
            waypoints = []
            for ii in range(i, end_i):
                for jj in range(j, end_j):
                    if ii != jj:
                        waypoints.append(f"{coords[ii][1]},{coords[ii][0]};{coords[jj][1]},{coords[jj][0]}")
            
            if waypoints:
                try:
                    # Make batch request
                    batch_url = f"{OSRM_URL}{';'.join(waypoints)}?overview=false"
                    response = requests.get(batch_url, timeout=15)
                    
                    if response.status_code == 200:
                        routes = response.json().get('routes', [])
                        route_idx = 0
                        
                        for ii in range(i, end_i):
                            for jj in range(j, end_j):
                                if ii != jj and route_idx < len(routes):
                                    matrix[ii][jj] = routes[route_idx]['distance'] / 1000
                                    route_idx += 1
                                elif ii != jj:
                                    # Fallback to straight-line distance
                                    matrix[ii][jj] = haversine_distance(
                                        coords[ii][0], coords[ii][1], 
                                        coords[jj][0], coords[jj][1]
                                    )
                    else:
                        # Fallback to straight-line distance for this batch
                        for ii in range(i, end_i):
                            for jj in range(j, end_j):
                                if ii != jj:
                                    matrix[ii][jj] = haversine_distance(
                                        coords[ii][0], coords[ii][1], 
                                        coords[jj][0], coords[jj][1]
                                    )
                except:
                    # Fallback to straight-line distance for this batch
                    for ii in range(i, end_i):
                        for jj in range(j, end_j):
                            if ii != jj:
                                matrix[ii][jj] = haversine_distance(
                                    coords[ii][0], coords[ii][1], 
                                    coords[jj][0], coords[jj][1]
                                )
    
    return matrix

def fast_tsp_solve(coords, distance_matrix):
    """Fast TSP solver using nearest neighbor heuristic"""
    n = len(coords)
    if n <= 2:
        return list(range(n))
    
    # Use nearest neighbor heuristic for speed
    unvisited = set(range(1, n))  # Start from index 0
    route = [0]
    current = 0
    
    while unvisited:
        # Find nearest unvisited neighbor
        nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return route

def optimize_clusters(df, n_clusters):
    """Optimize clustering with better initialization"""
    coords = df[['Latitude', 'Longitude']].to_numpy()
    
    # Use k-means++ initialization for better clustering
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=5,  # Reduced from 10 for speed
        random_state=42,
        init='k-means++',
        max_iter=100  # Reduced for speed
    )
    
    labels = kmeans.fit_predict(coords)
    return labels

def run_optimized_algorithm(df):
    """Run the optimized clustering-based algorithm"""
    try:
        print("Starting optimized algorithm...")
        start_time = time.time()
        
        # Prepare data
        df = df[['bus_stop_lat', 'bus_stop_lon', 'student_count']].copy()
        df.columns = ['Latitude', 'Longitude', 'StudentCount']
        
        # Skip accessibility check for speed (assume all points are accessible)
        print(f"Processing {len(df)} boarding points")
        
        coords = df[['Latitude', 'Longitude']].to_numpy()
        student_counts = df['StudentCount'].tolist()
        total_students = int(sum(student_counts))  # Convert to native Python int
        buses_needed = calculate_optimal_buses(total_students)
        
        print(f"Total students: {total_students}, Using {buses_needed} buses")

        # Optimized clustering
        print("Optimizing clusters...")
        labels = optimize_clusters(df, buses_needed)
        df['Cluster'] = labels

        # Solve TSP for each cluster with optimized approach
        print("Solving optimized TSP for each cluster...")
        all_routes = []
        total_distance = 0.0  # Use float for distance
        total_time = 0.0  # Use float for time
        
        for i in range(buses_needed):
            cluster_points = df[df['Cluster'] == i]
            if len(cluster_points) <= 1:
                continue
                
            cluster_coords = cluster_points[['Latitude', 'Longitude']].to_numpy()
            
            # Use fast distance calculation for small clusters
            if len(cluster_coords) <= 10:
                matrix = [[haversine_distance(cluster_coords[i][0], cluster_coords[i][1], 
                                            cluster_coords[j][0], cluster_coords[j][1]) 
                          for j in range(len(cluster_coords))] 
                         for i in range(len(cluster_coords))]
            else:
                matrix = batch_distance_matrix(cluster_coords)
            
            # Use fast TSP solver
            route = fast_tsp_solve(cluster_coords, matrix)
            
            if route:
                ordered_points = cluster_coords[route]
                route_stops = []
                
                # Add college as starting point
                route_stops.append({
                    'name': 'College',
                    'lat': float(COLLEGE_LAT),  # Convert to native Python float
                    'lon': float(COLLEGE_LON),  # Convert to native Python float
                    'students': 0,
                    'type': 'college'
                })
                
                # Add boarding points
                cluster_students = 0
                for j, point in enumerate(ordered_points):
                    original_idx = cluster_points.index[route[j]]
                    student_count = int(cluster_points.loc[original_idx, 'StudentCount'])  # Convert to native Python int
                    cluster_students += student_count
                    
                    route_stops.append({
                        'name': f"Route {i+1}",
                        'lat': float(point[0]),  # Convert to native Python float
                        'lon': float(point[1]),  # Convert to native Python float
                        'students': student_count,
                        'type': 'boarding_point'
                    })
                
                # Calculate route metrics
                route_distance = float(sum(matrix[route[j]][route[j+1]] for j in range(len(route)-1)))  # Convert to native Python float
                route_time = float(route_distance * 2)  # Convert to native Python float
                
                all_routes.append({
                    'bus_id': int(i + 1),  # Convert to native Python int
                    'stops': route_stops,
                    'total_students': int(cluster_students),  # Convert to native Python int
                    'total_distance_km': route_distance,
                    'total_time_min': route_time
                })
                
                total_distance += route_distance
                total_time += route_time

        elapsed_time = time.time() - start_time
        print(f"Generated {len(all_routes)} routes in {elapsed_time:.2f} seconds")
        
        return {
            "success": True,
            "routes": all_routes,
            "summary": {
                "total_routes": int(len(all_routes)),  # Convert to native Python int
                "total_students": total_students,
                "total_distance_km": float(total_distance),  # Convert to native Python float
                "total_time_min": float(total_time),  # Convert to native Python float
                "processing_time_seconds": float(elapsed_time)  # Convert to native Python float
            }
        }

    except Exception as e:
        print(f"Optimized algorithm error: {str(e)}")
        return {"error": f"Algorithm error: {str(e)}"}



def run_routing_algorithm(csv_data):
    """Run the routing algorithm on the provided CSV data"""
    try:
        # === Step 1: Load Data ===
        df = pd.read_csv(csv_data)
        
        # Validate required columns
        required_columns = ['cluster', 'student_count', 'bus_stop_lat', 'bus_stop_lon']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing required columns: {missing_columns}"}

        print(f"Loaded {len(df)} boarding points")
        
        # Use the new clustering-based algorithm
        return run_optimized_algorithm(df)

    except Exception as e:
        return {"error": f"Algorithm error: {str(e)}"}

    except Exception as e:
        return {"error": f"Algorithm error: {str(e)}"}

@app.route('/api/route', methods=['POST'])
def optimize_routes():
    """API endpoint to run the routing algorithm"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name

        try:
            # Run the routing algorithm
            result = run_routing_algorithm(tmp_file_path)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise e

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 