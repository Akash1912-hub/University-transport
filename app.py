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

app = Flask(__name__)
CORS(app)

# === Constants ===
COLLEGE_LAT = 13.0827  # Rajalakshmi Engineering College, Chennai
COLLEGE_LON = 80.2707
MAX_BUS_CAPACITY = 55
MAX_DISTANCE_KM = 40
MAX_DURATION_MIN = 120
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

def calculate_optimal_buses(total_students):
    return max(1, min(20, (total_students + MAX_BUS_CAPACITY - 1) // MAX_BUS_CAPACITY))

def is_accessible(lat, lon):
    """Check if a location is accessible using OSRM"""
    try:
        url = f"{OSRM_URL}{lon},{lat};{lon},{lat}?overview=false"
        r = requests.get(url, timeout=5)
        return r.status_code == 200 and r.json()['code'] == 'Ok'
    except:
        return False

def get_distance_matrix(coords):
    """Get distance matrix using OSRM API"""
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    
    print(f"Computing distance matrix for {n} points...")
    for i in range(n):
        print(f"Processing row {i+1}/{n}")
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                try:
                    url = f"{OSRM_URL}{coords[i][1]},{coords[i][0]};{coords[j][1]},{coords[j][0]}?overview=false"
                    res = requests.get(url, timeout=10).json()
                    if res.get("code") == "Ok":
                        matrix[i][j] = res["routes"][0]["distance"] / 1000  # km
                    else:
                        # Fallback to straight-line distance
                        import math
                        lat1, lon1 = coords[i][0], coords[i][1]
                        lat2, lon2 = coords[j][0], coords[j][1]
                        distance = math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
                        matrix[i][j] = distance
                except:
                    # Fallback to straight-line distance
                    import math
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

def run_new_algorithm(df):
    """Run the new clustering-based algorithm"""
    try:
        print("Starting new clustering algorithm...")
        
        # Prepare data
        df = df[['bus_stop_lat', 'bus_stop_lon', 'student_count']].copy()
        df.columns = ['Latitude', 'Longitude', 'StudentCount']
        
        # Remove inaccessible points
        print("Checking accessibility...")
        df['Accessible'] = df.apply(lambda x: is_accessible(x['Latitude'], x['Longitude']), axis=1)
        df = df[df['Accessible']]
        
        if df.empty:
            return {"error": "No accessible boarding points found"}

        coords = df[['Latitude', 'Longitude']].to_numpy()
        student_counts = df['StudentCount'].tolist()
        total_students = sum(student_counts)
        buses_needed = calculate_optimal_buses(total_students)
        
        print(f"Total students: {total_students}, Using {buses_needed} buses")

        # Cluster using KMeans
        print("Clustering boarding points...")
        kmeans = KMeans(n_clusters=buses_needed, n_init=10, random_state=42)
        labels = kmeans.fit_predict(coords)
        df['Cluster'] = labels

        # Solve TSP for each cluster
        print("Solving TSP for each cluster...")
        all_routes = []
        total_distance = 0
        total_time = 0
        
        for i in range(buses_needed):
            cluster_points = df[df['Cluster'] == i]
            if len(cluster_points) <= 1:
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
                    student_count = cluster_points.loc[original_idx, 'StudentCount']
                    cluster_students += student_count
                    
                    route_stops.append({
                        'name': f"Route {i+1}",
                        'lat': float(point[0]),
                        'lon': float(point[1]),
                        'students': int(student_count),
                        'type': 'boarding_point'
                    })
                
                # Calculate route metrics
                route_distance = sum(matrix[route[j]][route[j+1]] for j in range(len(route)-1))
                route_time = route_distance * 2  # 2 min per km estimate
                
                all_routes.append({
                    'bus_id': i + 1,
                    'stops': route_stops,
                    'total_students': cluster_students,
                    'total_distance_km': route_distance,
                    'total_time_min': route_time
                })
                
                total_distance += route_distance
                total_time += route_time

        print(f"Generated {len(all_routes)} routes")
        
        return {
            "success": True,
            "routes": all_routes,
            "summary": {
                "total_routes": len(all_routes),
                "total_students": total_students,
                "total_distance": total_distance,
                "total_time": total_time
            }
        }

    except Exception as e:
        print(f"New algorithm error: {str(e)}")
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
        return run_new_algorithm(df)

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