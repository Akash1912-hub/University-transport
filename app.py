from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import json
import tempfile
import os

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# === Constants ===
COLLEGE_LAT = 13.0827  # Rajalakshmi Engineering College, Chennai
COLLEGE_LON = 80.2707
MAX_BUS_CAPACITY = 55
MAX_DISTANCE_KM = 40
MAX_DURATION_MIN = 120
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

# Performance optimization settings
ENABLE_ROAD_CHECK = False  # Temporarily disabled for testing
ENABLE_DETAILED_ANALYSIS = False  # Temporarily disabled for testing
CLUSTERING_METHOD = 'kmeans'
MAX_CLUSTERS = 20
MIN_CLUSTER_SIZE = 3

def calculate_optimal_buses(total_students):
    """Calculate optimal number of buses needed"""
    return max(1, min(20, (total_students + MAX_BUS_CAPACITY - 1) // MAX_BUS_CAPACITY))

def is_accessible(lat, lon):
    """Check if a location is accessible by heavy vehicles"""
    try:
        # Check geographic bounds first
        if not (12.5 <= lat <= 13.5 and 79.5 <= lon <= 81.0):
            print(f"❌ Point ({lat:.4f}, {lon:.4f}) outside Chennai bounds")
            return False
            
        # If road check is disabled, accept all points within bounds
        if not ENABLE_ROAD_CHECK:
            print(f"✅ Point ({lat:.4f}, {lon:.4f}) accepted (road check disabled)")
            return True
            
        # Test route from college to this point
        url = f"{OSRM_URL}{COLLEGE_LON},{COLLEGE_LAT};{lon},{lat}?overview=false&profile=driving"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                distance = route['distance'] / 1000  # Convert to km
                
                if distance > 0 and distance < MAX_DISTANCE_KM:
                    print(f"✅ Point ({lat:.4f}, {lon:.4f}) accessible: {distance:.2f} km")
                    return True
                else:
                    print(f"❌ Point ({lat:.4f}, {lon:.4f}) too far: {distance:.2f} km")
                    return False
            else:
                print(f"❌ Point ({lat:.4f}, {lon:.4f}) no route found")
                return False
        else:
            print(f"❌ Point ({lat:.4f}, {lon:.4f}) API error: {response.status_code}")
            # Fallback: accept point if API is down
            print(f"⚠️ Accepting point ({lat:.4f}, {lon:.4f}) as fallback (API error)")
            return True
            
    except Exception as e:
        print(f"❌ Point ({lat:.4f}, {lon:.4f}) accessibility check failed: {str(e)}")
        # Fallback: accept point if exception occurs
        print(f"⚠️ Accepting point ({lat:.4f}, {lon:.4f}) as fallback (exception)")
        return True

def get_distance_matrix(coords):
    """Get distance matrix using road distances"""
    import math
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    
    print(f"Computing distance matrix for {n} points...")
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                # Use straight-line distance for speed
                lat1, lon1 = coords[i][0], coords[i][1]
                lat2, lon2 = coords[j][0], coords[j][1]
                distance = math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
                matrix[i][j] = distance
    
    return matrix

def solve_tsp(distance_matrix):
    """Solve Traveling Salesman Problem using OR-Tools"""
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_i, to_i):
        return int(distance_matrix[manager.IndexToNode(from_i)][manager.IndexToNode(to_i)] * 1000)

    transit_cb_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return []

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    return route

def perform_clustering(coords, student_counts, total_students):
    """Perform clustering using KMeans or DBSCAN"""
    print(f"🔄 Performing {CLUSTERING_METHOD.upper()} clustering...")
    
    if CLUSTERING_METHOD == 'kmeans':
        buses_needed = calculate_optimal_buses(total_students)
        buses_needed = min(buses_needed, MAX_CLUSTERS)
        
        print(f"📊 Using KMeans with {buses_needed} clusters")
        kmeans = KMeans(n_clusters=buses_needed, n_init=10, random_state=42)
        labels = kmeans.fit_predict(coords)
        
    elif CLUSTERING_METHOD == 'dbscan':
        print("📊 Using DBSCAN clustering")
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        eps = 0.5
        dbscan = DBSCAN(eps=eps, min_samples=MIN_CLUSTER_SIZE)
        labels = dbscan.fit_predict(coords_scaled)
        
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        buses_needed = len(unique_labels)
        
        print(f"📊 DBSCAN found {buses_needed} clusters")
    
    else:
        raise ValueError(f"Unknown clustering method: {CLUSTERING_METHOD}")
    
    return labels, buses_needed

def run_routing_algorithm(df):
    """Run the routing algorithm on the provided data"""
    try:
        print("Starting routing algorithm...")
        
        # Prepare data
        df = df[['bus_stop_lat', 'bus_stop_lon', 'student_count']].copy()
        df.columns = ['Latitude', 'Longitude', 'StudentCount']
        
        # Check accessibility
        print("Checking accessibility...")
        if ENABLE_ROAD_CHECK:
            df['Accessible'] = df.apply(lambda x: is_accessible(x['Latitude'], x['Longitude']), axis=1)
            accessible_count = df['Accessible'].sum()
            print(f"Found {accessible_count} accessible points out of {len(df)} total")
            df = df[df['Accessible']]
        else:
            print("⚠️ Road accessibility check disabled for faster processing")
            df['Accessible'] = True
            accessible_count = len(df)
        
        if df.empty:
            print("❌ No accessible boarding points found")
            return {"error": "No accessible boarding points found. Please check coordinates."}

        coords = df[['Latitude', 'Longitude']].to_numpy()
        student_counts = df['StudentCount'].tolist()
        total_students = int(sum(student_counts))
        
        print(f"Total students: {total_students}")

        # Perform clustering
        labels, buses_needed = perform_clustering(coords, student_counts, total_students)
        df['Cluster'] = labels

        # Solve TSP for each cluster
        print(f"Solving TSP for {buses_needed} clusters...")
        all_routes = []
        total_distance = 0
        total_time = 0
        total_students_served = 0
        
        for i in range(buses_needed):
            cluster_points = df[df['Cluster'] == i]
            print(f"Processing cluster {i+1}/{buses_needed} with {len(cluster_points)} points")
            
            if len(cluster_points) <= 1:
                print(f"Skipping cluster {i+1} - only {len(cluster_points)} points")
                continue
                
            cluster_coords = cluster_points[['Latitude', 'Longitude']].to_numpy()
            matrix = get_distance_matrix(cluster_coords)
            route = solve_tsp(matrix)
            
            if route:
                ordered_points = cluster_coords[route]
                route_stops = []
                
                # Add college as starting point
                route_stops.append({
                    'name': 'College',
                    'lat': COLLEGE_LAT,
                    'lon': COLLEGE_LON,
                    'students': 0,
                    'type': 'college'
                })
                
                # Add boarding points
                cluster_students = 0
                for j, point in enumerate(ordered_points):
                    original_idx = cluster_points.index[route[j]]
                    student_count = int(cluster_points.loc[original_idx, 'StudentCount'])
                    cluster_students += student_count
                    
                    route_stops.append({
                        'name': f"Route {i+1}",
                        'lat': float(point[0]),
                        'lon': float(point[1]),
                        'students': int(student_count),
                        'type': 'boarding_point'
                    })
                
                # Calculate route metrics correctly
                route_distance = 0
                if len(route) > 1:
                    # Calculate distance from college to first point
                    college_to_first = np.sqrt((ordered_points[0][0] - COLLEGE_LAT)**2 + 
                                            (ordered_points[0][1] - COLLEGE_LON)**2) * 111
                    route_distance += college_to_first
                    
                    # Calculate distances between points
                    for j in range(len(route) - 1):
                        point1 = ordered_points[j]
                        point2 = ordered_points[j + 1]
                        distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) * 111
                        route_distance += distance
                    
                    # Add distance back to college
                    last_to_college = np.sqrt((ordered_points[-1][0] - COLLEGE_LAT)**2 + 
                                           (ordered_points[-1][1] - COLLEGE_LON)**2) * 111
                    route_distance += last_to_college
                
                route_time = route_distance * 2  # 2 min per km estimate
                
                # Check constraints
                constraint_violations = []
                
                if cluster_students > MAX_BUS_CAPACITY:
                    constraint_violations.append(f"Capacity exceeded: {cluster_students} > {MAX_BUS_CAPACITY}")
                
                if route_distance > MAX_DISTANCE_KM:
                    constraint_violations.append(f"Distance exceeded: {route_distance:.2f} > {MAX_DISTANCE_KM}")
                
                if route_time > MAX_DURATION_MIN:
                    constraint_violations.append(f"Time exceeded: {route_time:.1f} > {MAX_DURATION_MIN}")
                
                # Add route with constraint information
                route_data = {
                    'bus_id': i + 1,
                    'stops': route_stops,
                    'total_students': int(cluster_students),
                    'total_distance_km': float(route_distance),
                    'total_time_min': float(route_time),
                    'constraint_violations': constraint_violations,
                    'is_feasible': len(constraint_violations) == 0
                }
                
                all_routes.append(route_data)
                total_distance += route_distance
                total_time += route_time
                total_students_served += cluster_students
                
                if constraint_violations:
                    print(f"⚠️ Route {i+1} has constraint violations: {constraint_violations}")
                else:
                    print(f"✅ Route {i+1} meets all constraints")

        print(f"✅ Generated {len(all_routes)} routes successfully!")
        print(f"📊 Summary: {total_students_served} students served, {total_distance:.2f} km total distance")
        
        # Calculate constraint statistics
        feasible_routes = [r for r in all_routes if r['is_feasible']]
        infeasible_routes = [r for r in all_routes if not r['is_feasible']]
        
        print(f"📊 Constraint Analysis:")
        print(f"   ✅ Feasible routes: {len(feasible_routes)}")
        print(f"   ❌ Infeasible routes: {len(infeasible_routes)}")
        
        return {
            "success": True,
            "routes": all_routes,
            "summary": {
                "total_routes": int(len(all_routes)),
                "buses_required": int(buses_needed),
                "feasible_routes": int(len(feasible_routes)),
                "infeasible_routes": int(len(infeasible_routes)),
                "total_students": int(total_students),
                "students_served": int(total_students_served),
                "total_distance": float(total_distance),
                "total_time": float(total_time),
                "constraints": {
                    "max_bus_capacity": MAX_BUS_CAPACITY,
                    "max_distance_km": MAX_DISTANCE_KM,
                    "max_duration_min": MAX_DURATION_MIN
                },
                "algorithm_settings": {
                    "clustering_method": CLUSTERING_METHOD,
                    "road_check_enabled": ENABLE_ROAD_CHECK,
                    "detailed_analysis": ENABLE_DETAILED_ANALYSIS
                }
            }
        }

    except Exception as e:
        print(f"Algorithm error: {str(e)}")
        return {"error": f"Algorithm error: {str(e)}"}

@app.route('/api/route', methods=['POST'])
def optimize_routes():
    """API endpoint to run the routing algorithm"""
    try:
        print("=== File Upload Request Received ===")
        print(f"Request files: {list(request.files.keys())}")
        
        # Check if file is present
        if 'file' not in request.files:
            print("❌ No file in request.files")
            return jsonify({"error": "No file provided. Please select a CSV file."}), 400
        
        file = request.files['file']
        print(f"📁 File received: {file.filename}")
        
        # Check if filename is empty
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({"error": "No file selected. Please choose a CSV file."}), 400
        
        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            print("❌ Invalid file type")
            return jsonify({"error": "Please upload a CSV file."}), 400
        
        print("✅ File validation passed:", file.filename)
        
        # Save file temporarily
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
                file.save(tmp_file.name)
                tmp_file_path = tmp_file.name
                print(f"💾 File saved temporarily: {tmp_file_path}")
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            return jsonify({"error": f"Error saving file: {str(e)}"}), 500
        
        try:
            # Load and validate CSV
            print("🚀 Starting routing algorithm...")
            df = pd.read_csv(tmp_file_path)
            
            print(f"CSV loaded successfully with {len(df)} rows")
            print(f"Columns found: {list(df.columns)}")
            
            # Validate required columns
            required_columns = ['cluster', 'student_count', 'bus_stop_lat', 'bus_stop_lon']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({"error": f"Missing required columns: {missing_columns}. Found columns: {list(df.columns)}"}), 400

            print(f"All required columns present. Processing {len(df)} boarding points")
            
            # Run the routing algorithm
            result = run_routing_algorithm(df)
            print("✅ Algorithm completed. Result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Algorithm error: {str(e)}")
            return jsonify({"error": f"Algorithm error: {str(e)}"}), 500
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
                print("🧹 Temporary file cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning up temporary file: {str(e)}")
                
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Transport optimization API is running"})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for debugging"""
    return jsonify({"message": "Test endpoint working", "settings": {
        "road_check_enabled": ENABLE_ROAD_CHECK,
        "detailed_analysis": ENABLE_DETAILED_ANALYSIS,
        "clustering_method": CLUSTERING_METHOD
    }})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 