from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import openrouteservice
import polyline
import tempfile
import os
import time
import math

app = Flask(__name__)
CORS(app)

# === Constants ===
COLLEGE_LAT = 13.0827  # Rajalakshmi Engineering College, Chennai
COLLEGE_LON = 80.2707
MAX_BUS_CAPACITY = 55
MAX_DISTANCE_KM = 40
MAX_DURATION_MIN = 120
# You'll need to add your OpenRouteService API key here
ORS_API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQ4MjA0YTNjOTE5MTRjMTE5M2QzZjg1YzM1YWQxYjQwIiwiaCI6Im11cm11cjY0In0='  # Replace with actual key

def load_boarding_points(df):
    """Load and prepare boarding points data"""
    # Check if required columns exist
    required_columns = ['bus_stop_lat', 'bus_stop_lon', 'student_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with missing values
    df = df.dropna(subset=['bus_stop_lat', 'bus_stop_lon', 'student_count'])
    
    # Check if we have any data left
    if len(df) == 0:
        raise ValueError("No valid data points found after removing missing values")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'bus_stop_lat': 'latitude',
        'bus_stop_lon': 'longitude', 
        'student_count': 'no_of_students'
    })
    
    # Ensure numeric types
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['no_of_students'] = pd.to_numeric(df['no_of_students'], errors='coerce')
    
    # Remove rows with invalid coordinates or student counts
    df = df.dropna(subset=['latitude', 'longitude', 'no_of_students'])
    
    # Check if we have any data left
    if len(df) == 0:
        raise ValueError("No valid data points found after data validation")
    
    print(f"Loaded {len(df)} valid boarding points")
    return df

def check_accessibility(df):
    """Road accessibility check using OSMNX"""
    try:
        # Create graph around the mean location
        center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
        
        # Check if coordinates are valid
        if pd.isna(center_lat) or pd.isna(center_lon):
            print("Invalid coordinates detected, skipping accessibility check")
            return df
        
        G = ox.graph_from_point((center_lat, center_lon), dist=10000, network_type='drive')
        
        accessible = []
        for _, row in df.iterrows():
            try:
                # Check if coordinates are valid
                if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                    accessible.append(False)
                    continue
                    
                ox.get_nearest_node(G, (row['latitude'], row['longitude']))
                accessible.append(True)
            except Exception as e:
                print(f"Point accessibility check failed: {e}")
                accessible.append(False)
        
        df['accessible'] = accessible
        accessible_df = df[df['accessible'] == True]
        
        if len(accessible_df) == 0:
            print("No accessible points found, returning all points")
            return df
        
        return accessible_df
    except Exception as e:
        print(f"Accessibility check failed: {e}")
        # Return all points if accessibility check fails
        return df

def cluster_points(df, college_coord, max_students=55):
    """Capacity-Constrained Clustering"""
    # Check if DataFrame is empty
    if len(df) == 0:
        print("Warning: No data points available for clustering")
        return df
    
    # Ensure we have at least one cluster
    num_clusters = max(1, len(df) // max(1, (max_students // df['no_of_students'].max())))
    
    # If only one point, create a single cluster
    if len(df) == 1:
        df['cluster'] = 0
        return df
    
    coords = df[['latitude', 'longitude']].values
    
    # Ensure we don't have more clusters than points
    num_clusters = min(num_clusters, len(df))
    
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
        df['cluster'] = kmeans.labels_
    except Exception as e:
        print(f"KMeans clustering failed: {e}")
        # Fallback: assign all points to cluster 0
        df['cluster'] = 0
        return df

    # Fix clusters exceeding capacity
    final_clusters = []
    for cluster_id in range(num_clusters):
        cluster = df[df['cluster'] == cluster_id]
        if len(cluster) == 0:
            continue
            
        if cluster['no_of_students'].sum() > max_students:
            # Split large clusters
            n_splits = int(np.ceil(cluster['no_of_students'].sum() / max_students))
            split = np.array_split(cluster, n_splits)
            for i, s in enumerate(split):
                if len(s) > 0:  # Only add non-empty splits
                    s['cluster'] = f"{cluster_id}_{i}"
                    final_clusters.append(s)
        else:
            final_clusters.append(cluster)
    
    if not final_clusters:
        # If no clusters were created, return original data with single cluster
        df['cluster'] = 0
        return df
    
    result = pd.concat(final_clusters)
    return result

def get_distance_matrix(cluster_df, college_coord, api_key):
    """Get OSRM Distance Matrix using OpenRouteService"""
    try:
        client = openrouteservice.Client(key=api_key)
        locations = [(college_coord[1], college_coord[0])] + list(zip(cluster_df['longitude'], cluster_df['latitude']))
        res = client.distance_matrix(locations=locations, profile='driving-car', metrics=['distance', 'duration'])
        return res['distances'], res['durations']
    except Exception as e:
        print(f"Distance matrix calculation failed: {e}")
        # Fallback to simple distance calculation
        return fallback_distance_matrix(cluster_df, college_coord)

def fallback_distance_matrix(cluster_df, college_coord):
    """Fallback distance calculation when API fails"""
    n = len(cluster_df) + 1  # +1 for college
    distances = [[0] * n for _ in range(n)]
    durations = [[0] * n for _ in range(n)]
    
    # Calculate distances using simple formula
    for i in range(n):
        for j in range(n):
            if i != j:
                if i == 0:  # College
                    lat1, lon1 = college_coord[0], college_coord[1]
                else:
                    lat1, lon1 = cluster_df.iloc[i-1]['latitude'], cluster_df.iloc[i-1]['longitude']
                
                if j == 0:  # College
                    lat2, lon2 = college_coord[0], college_coord[1]
                else:
                    lat2, lon2 = cluster_df.iloc[j-1]['latitude'], cluster_df.iloc[j-1]['longitude']
                
                # Simple distance calculation
                dx = (lon2 - lon1) * 111 * math.cos(math.radians((lat1 + lat2) / 2))
                dy = (lat2 - lat1) * 111
                distance = math.sqrt(dx * dx + dy * dy)
                
                distances[i][j] = int(distance * 1000)  # Convert to meters
                durations[i][j] = int(distance * 2 * 60)  # 2 min per km
    
    return distances, durations

def solve_tsp(distance_matrix):
    """Solve TSP with OR-Tools"""
    tsp_size = len(distance_matrix)
    routing = pywrapcp.RoutingModel(tsp_size, 1, 0)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.time_limit.seconds = 10  # 10 second limit

    def distance_callback(from_node, to_node):
        return int(distance_matrix[from_node][to_node])

    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
    assignment = routing.SolveWithParameters(search_parameters)

    route = []
    if assignment:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(routing.IndexToNode(index))
            index = assignment.Value(routing.NextVar(index))
        route.append(routing.IndexToNode(index))
    return route

def build_polyline(route_coords, api_key):
    """Build polyline for frontend map display"""
    try:
        client = openrouteservice.Client(key=api_key)
        geometry = client.directions(route_coords, profile='driving-car')['routes'][0]['geometry']
        return polyline.decode(geometry)
    except Exception as e:
        print(f"Polyline generation failed: {e}")
        # Return simple straight line
        return route_coords

def run_optimized_algorithm(df):
    """Run the optimized routing algorithm"""
    try:
        print("Starting optimized algorithm...")
        start_time = time.time()
        
        # Prepare data
        df = load_boarding_points(df)
        print(f"Processing {len(df)} boarding points")
        
        # Validate data
        if len(df) == 0:
            return {"error": "No data points found in the uploaded file"}
        
        # Check accessibility
        df = check_accessibility(df)
        print(f"After accessibility check: {len(df)} points")
        
        # Check if we still have data after accessibility filtering
        if len(df) == 0:
            return {"error": "No accessible boarding points found. Please check your coordinates."}
        
        # Cluster points
        college_coord = (COLLEGE_LAT, COLLEGE_LON)
        df = cluster_points(df, college_coord, MAX_BUS_CAPACITY)
        print(f"Clustered into {df['cluster'].nunique()} clusters")
        
        # Check if clustering was successful
        if len(df) == 0:
            return {"error": "Clustering failed. Please check your data."}
        
        # Process each cluster
        all_routes = []
        total_distance = 0.0
        total_time = 0.0
        
        for cluster_id in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster_id]
            print(f"Processing cluster {cluster_id} with {len(cluster_df)} points")
            
            # Get distance matrix
            distance_matrix, duration_matrix = get_distance_matrix(cluster_df, college_coord, ORS_API_KEY)
            
            # Solve TSP
            route_indices = solve_tsp(distance_matrix)
            
            if not route_indices:
                print(f"No solution found for cluster {cluster_id}")
                continue
            
            # Build route coordinates
            ordered_coords = [(college_coord[1], college_coord[0])] + list(zip(cluster_df['longitude'], cluster_df['latitude']))
            route_coords = [ordered_coords[i] for i in route_indices]
            
            # Calculate metrics
            total_duration = sum(duration_matrix[route_indices[i]][route_indices[i+1]] for i in range(len(route_indices)-1)) / 60
            total_distance_km = sum(distance_matrix[route_indices[i]][route_indices[i+1]] for i in range(len(route_indices)-1)) / 1000
            
            # Check constraints
            if total_duration <= MAX_DURATION_MIN and total_distance_km <= MAX_DISTANCE_KM:
                # Build route stops
                route_stops = []
                
                # Add college as starting point
                route_stops.append({
                    'name': 'College',
                    'lat': float(COLLEGE_LAT),
                    'lon': float(COLLEGE_LON),
                    'students': 0,
                    'type': 'college'
                })
                
                # Add boarding points
                cluster_students = 0
                for i, coord in enumerate(route_coords[1:], 1):  # Skip college
                    student_count = int(cluster_df.iloc[route_indices[i]-1]['no_of_students'])
                    cluster_students += student_count
                    
                    route_stops.append({
                        'name': f"Route {cluster_id}",
                        'lat': float(coord[1]),
                        'lon': float(coord[0]),
                        'students': student_count,
                        'type': 'boarding_point'
                    })
                
                all_routes.append({
                    'bus_id': int(len(all_routes) + 1),
                    'stops': route_stops,
                    'total_students': int(cluster_students),
                    'total_distance_km': float(total_distance_km),
                    'total_time_min': float(total_duration)
                })
                
                total_distance += total_distance_km
                total_time += total_duration
                
                print(f"Route {cluster_id}: {len(route_stops)} stops, {total_distance_km:.1f} km, {total_duration:.0f} min")
            else:
                print(f"Route {cluster_id} exceeds time/distance limits")
        
        elapsed_time = time.time() - start_time
        print(f"Generated {len(all_routes)} routes in {elapsed_time:.2f} seconds")
        
        return {
            "success": True,
            "routes": all_routes,
            "summary": {
                "total_routes": int(len(all_routes)),
                "total_students": int(sum(route['total_students'] for route in all_routes)),
                "total_distance_km": float(total_distance),
                "total_time_min": float(total_time),
                "processing_time_seconds": float(elapsed_time)
            }
        }
        
    except Exception as e:
        print(f"Algorithm error: {str(e)}")
        return {"error": f"Algorithm error: {str(e)}"}

def run_routing_algorithm(csv_data):
    """Run the routing algorithm on the provided CSV data"""
    try:
        # Load data
        df = pd.read_csv(csv_data)
        
        # Validate required columns
        required_columns = ['cluster', 'student_count', 'bus_stop_lat', 'bus_stop_lon']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing required columns: {missing_columns}"}

        print(f"Loaded {len(df)} boarding points")
        
        # Run the optimized algorithm
        return run_optimized_algorithm(df)

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