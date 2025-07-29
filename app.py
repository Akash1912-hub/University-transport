from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import osmnx as ox
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import json
import tempfile
import os

app = Flask(__name__)
CORS(app)

# === Constants ===
COLLEGE_LAT = 13.0827  # Rajalakshmi Engineering College, Chennai
COLLEGE_LON = 80.2707
MAX_STUDENTS_PER_BUS = 55
MAX_ROUTE_DISTANCE_KM = 40
MAX_ROUTE_TIME_MIN = 120
# Calculate optimal number of buses based on total students
def calculate_optimal_buses(total_students):
    return max(1, min(20, (total_students + MAX_STUDENTS_PER_BUS - 1) // MAX_STUDENTS_PER_BUS))

# Performance optimization constants
MAX_BOARDING_POINTS_FOR_EXACT = 1000  # Always use exact algorithm (OR-Tools)
CHUNK_SIZE = 25  # Process boarding points in chunks for large datasets

def run_exact_algorithm(df, G, locations, student_counts, node_ids, num_buses):
    """Run the exact algorithm for small datasets (≤50 boarding points)"""
    try:
        print("Starting exact algorithm...")
        
        # === Step 4: Compute Distance and Time Matrices ===
        print("Computing distance matrix...")
        distance_matrix = []
        time_matrix = []
        
        n = len(node_ids)
        for i in range(n):
            print(f"Processing row {i+1}/{n}")
            d_row, t_row = [], []
            for j in range(n):
                try:
                    length = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight='length')
                    time = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight='travel_time')
                    d_row.append(length / 1000)
                    t_row.append(time / 60)
                except:
                    # Use straight-line distance as fallback
                    import math
                    if i == 0:  # College
                        lat1, lon1 = COLLEGE_LAT, COLLEGE_LON
                    else:
                        lat1, lon1 = df.iloc[i-1]['bus_stop_lat'], df.iloc[i-1]['bus_stop_lon']
                    
                    if j == 0:  # College
                        lat2, lon2 = COLLEGE_LAT, COLLEGE_LON
                    else:
                        lat2, lon2 = df.iloc[j-1]['bus_stop_lat'], df.iloc[j-1]['bus_stop_lon']
                    
                    distance = math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
                    d_row.append(distance)
                    t_row.append(distance * 2)  # 2 min per km estimate
            distance_matrix.append(d_row)
            time_matrix.append(t_row)

        print("Distance matrix computed successfully")

        # === Step 5: OR-Tools Routing Solver ===
        print("Setting up OR-Tools solver...")
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_buses, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx, to_idx):
            return int(distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] * 1000)

        def time_callback(from_idx, to_idx):
            return int(time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])

        # Register callbacks
        routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))

        demand_callback_index = routing.RegisterUnaryTransitCallback(lambda i: student_counts[manager.IndexToNode(i)])
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, [MAX_STUDENTS_PER_BUS] * num_buses, True, "Capacity"
        )

        routing.AddDimension(
            routing.RegisterTransitCallback(distance_callback), 0, MAX_ROUTE_DISTANCE_KM * 1000, True, "Distance"
        )

        routing.AddDimension(
            routing.RegisterTransitCallback(time_callback), 0, MAX_ROUTE_TIME_MIN, True, "Time"
        )

        # Solve with optimized parameters
        print("Solving with OR-Tools...")
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.time_limit.seconds = 60  # Increased time limit for OR-Tools
        search_params.log_search = True
        solution = routing.SolveWithParameters(search_params)

        print(f"OR-Tools solution status: {solution}")
        if solution:
            print("✅ OR-Tools found a solution!")
        else:
            print("❌ OR-Tools failed to find solution")

        # === Step 6: Output Routes ===
        routes = []
        if solution:
            print("Extracting routes from solution...")
            for vehicle_id in range(num_buses):
                index = routing.Start(vehicle_id)
                route = []
                load, total_dist, total_time = 0, 0, 0
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(node)
                    load += student_counts[node]
                    prev_index = index
                    index = solution.Value(routing.NextVar(index))
                    total_dist += distance_callback(prev_index, index)
                    total_time += time_callback(prev_index, index)
                if len(route) > 1:
                    routes.append({
                        'bus': vehicle_id + 1,
                        'route': route,
                        'students': load,
                        'distance_km': total_dist / 1000,
                        'travel_time_min': total_time
                    })
        
        if not routes:
            print("OR-Tools failed, falling back to greedy algorithm...")
            return run_chunked_algorithm(df, G, num_buses)

        print(f"Generated {len(routes)} routes")
        return prepare_route_response(routes, df)

    except Exception as e:
        print(f"Exact algorithm error: {str(e)}")
        print("Falling back to greedy algorithm...")
        return run_chunked_algorithm(df, G, num_buses)

def run_chunked_algorithm(df, G, num_buses):
    """Run the chunked algorithm for large datasets (>50 boarding points)"""
    try:
        print("Starting chunked algorithm...")
        
        # Sort boarding points by student count (descending) for better distribution
        df_sorted = df.sort_values('student_count', ascending=False).reset_index(drop=True)
        
        print(f"Processing {len(df_sorted)} boarding points with {num_buses} buses")
        
        routes = []
        current_bus = 1
        current_students = 0
        current_route = []
        
        # Use greedy approach for large datasets
        for idx, row in df_sorted.iterrows():
            if current_students + row['student_count'] <= MAX_STUDENTS_PER_BUS:
                # Add to current bus
                current_route.append({
                    'lat': row['bus_stop_lat'],
                    'lon': row['bus_stop_lon'],
                    'students': row['student_count'],
                    'cluster': row['cluster']
                })
                current_students += row['student_count']
            else:
                # Current bus is full, start new bus
                if current_route:
                    routes.append({
                        'bus': current_bus,
                        'route': current_route,
                        'students': current_students,
                        'distance_km': calculate_route_distance_fast(current_route),
                        'travel_time_min': calculate_route_time_fast(current_route)
                    })
                    current_bus += 1
                    current_route = [{
                        'lat': row['bus_stop_lat'],
                        'lon': row['bus_stop_lon'],
                        'students': row['student_count'],
                        'cluster': row['cluster']
                    }]
                    current_students = row['student_count']
        
        # Add the last route
        if current_route:
            routes.append({
                'bus': current_bus,
                'route': current_route,
                'students': current_students,
                'distance_km': calculate_route_distance_fast(current_route),
                'travel_time_min': calculate_route_time_fast(current_route)
            })
        
        print(f"Generated {len(routes)} routes successfully")
        return prepare_route_response(routes, df_sorted)
        
    except Exception as e:
        print(f"Chunked algorithm error: {str(e)}")
        return {"error": f"Chunked algorithm error: {str(e)}"}

def calculate_route_distance_fast(route):
    """Calculate total distance for a route using straight-line distance (fast)"""
    if len(route) < 2:
        return 0
    
    import math
    total_distance = 0
    prev_lat, prev_lon = COLLEGE_LAT, COLLEGE_LON
    
    for stop in route:
        current_lat, current_lon = stop['lat'], stop['lon']
        distance = math.sqrt((current_lat-prev_lat)**2 + (current_lon-prev_lon)**2) * 111  # Rough km conversion
        total_distance += distance
        prev_lat, prev_lon = current_lat, current_lon
    
    return total_distance

def calculate_route_time_fast(route):
    """Calculate total time for a route using distance-based estimation (fast)"""
    if len(route) < 2:
        return 0
    
    distance = calculate_route_distance_fast(route)
    return distance * 2  # Rough estimate: 2 min per km

def calculate_route_distance(route, G):
    """Calculate total distance for a route (original method with graph)"""
    if len(route) < 2:
        return 0
    
    total_distance = 0
    prev_node = ox.distance.nearest_nodes(G, COLLEGE_LON, COLLEGE_LAT)
    
    for stop in route:
        current_node = ox.distance.nearest_nodes(G, stop['lon'], stop['lat'])
        try:
            distance = nx.shortest_path_length(G, prev_node, current_node, weight='length')
            total_distance += distance / 1000  # Convert to km
        except:
            # If path not found, use straight-line distance
            import math
            lat1, lon1 = COLLEGE_LAT, COLLEGE_LON
            lat2, lon2 = stop['lat'], stop['lon']
            distance = math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111  # Rough km conversion
            total_distance += distance
        prev_node = current_node
    
    return total_distance

def calculate_route_time(route, G):
    """Calculate total time for a route (original method with graph)"""
    if len(route) < 2:
        return 0
    
    total_time = 0
    prev_node = ox.distance.nearest_nodes(G, COLLEGE_LON, COLLEGE_LAT)
    
    for stop in route:
        current_node = ox.distance.nearest_nodes(G, stop['lon'], stop['lat'])
        try:
            time = nx.shortest_path_length(G, prev_node, current_node, weight='travel_time')
            total_time += time / 60  # Convert to minutes
        except:
            # If path not found, estimate time based on distance
            import math
            lat1, lon1 = COLLEGE_LAT, COLLEGE_LON
            lat2, lon2 = stop['lat'], stop['lon']
            distance = math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
            total_time += distance * 2  # Rough estimate: 2 min per km
        prev_node = current_node
    
    return total_time

def prepare_route_response(routes, df):
    """Prepare the final route response"""
    route_details = []
    for route in routes:
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
        for stop in route['route']:
            route_stops.append({
                'name': f"Route {stop['cluster']}",
                'lat': stop['lat'],
                'lon': stop['lon'],
                'students': stop['students'],
                'type': 'boarding_point'
            })
        
        route_details.append({
            'bus_id': route['bus'],
            'stops': route_stops,
            'total_students': route['students'],
            'total_distance_km': route['distance_km'],
            'total_time_min': route['travel_time_min']
        })

    return {
        "success": True,
        "routes": route_details,
        "summary": {
            "total_routes": len(routes),
            "total_students": sum(route['students'] for route in routes),
            "total_distance": sum(route['distance_km'] for route in routes),
            "total_time": sum(route['travel_time_min'] for route in routes)
        }
    }

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

        # === Step 2: Remove Inaccessible Stops ===
        # Use smaller distance for faster processing
        G = ox.graph_from_point((COLLEGE_LAT, COLLEGE_LON), dist=30000, network_type='drive_service')

        def is_accessible(lat, lon):
            try:
                ox.distance.nearest_nodes(G, lon, lat)
                return True
            except:
                return False

        df['is_accessible'] = df.apply(lambda row: is_accessible(row['bus_stop_lat'], row['bus_stop_lon']), axis=1)
        df = df[df['is_accessible'] == True].reset_index(drop=True)

        if df.empty:
            return {"error": "No accessible boarding points found"}

        # === Step 3: Prepare Locations ===
        locations = [(COLLEGE_LAT, COLLEGE_LON)] + list(zip(df['bus_stop_lat'], df['bus_stop_lon']))
        student_counts = [0] + list(df['student_count'])
        node_ids = [ox.distance.nearest_nodes(G, lon, lat) for lat, lon in locations]

        # Calculate optimal number of buses
        total_students = sum(df['student_count'])
        num_buses = calculate_optimal_buses(total_students)
        
        print(f"Total students: {total_students}, Using {num_buses} buses")
        print(f"Total boarding points: {len(df)}")
        
        # Performance optimization: Use different strategies based on dataset size
        if len(df) <= MAX_BOARDING_POINTS_FOR_EXACT:
            print("Using exact algorithm for small dataset")
            result = run_exact_algorithm(df, G, locations, student_counts, node_ids, num_buses)
            if "error" in result:
                print("Exact algorithm failed, trying chunked algorithm...")
                return run_chunked_algorithm(df, G, num_buses)
            return result
        else:
            print("Using chunked algorithm for large dataset")
            return run_chunked_algorithm(df, G, num_buses)

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