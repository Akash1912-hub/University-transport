from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import requests
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
import logging
from io import StringIO
import polyline
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# === Configuration ===
COLLEGE_LAT = 13.0827  # Rajalakshmi Engineering College, Chennai
COLLEGE_LON = 80.2707
MAX_BUS_CAPACITY = 55
MAX_DISTANCE_KM = 40
MAX_DURATION_HOURS = 2.0
GOOGLE_API_KEY = "AIzaSyBlfqs5K9HEe9c1Eu5bjPXXjr8Hz2mbTZE"  # Use your Google API key

@dataclass
class BoardingPoint:
    cluster_id: str
    student_count: int
    latitude: float
    longitude: float
    accessible: bool = True

@dataclass
class OptimizedRoute:
    bus_id: int
    cluster_id: str
    total_students: int
    stops: List[Dict]
    total_distance_km: float
    total_duration_hours: float
    polyline: str
    accessible: bool

class GoogleBasedRouteOptimizer:
    def __init__(self, college_lat: float, college_lon: float, google_api_key: str):
        self.college_lat = college_lat
        self.college_lon = college_lon
        self.google_api_key = google_api_key
        self.boarding_points = []
        self.optimized_routes = []
        
    def load_boarding_points(self, df: pd.DataFrame) -> List[BoardingPoint]:
        """Load boarding points from CSV"""
        try:
            # Validate required columns
            required_columns = ['cluster', 'student_count', 'bus_stop_lat', 'bus_stop_lon']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and validate data
            df = df.dropna(subset=required_columns)
            
            if len(df) == 0:
                raise ValueError("No valid data points found")
            
            # Convert to numeric with better error handling
            numeric_columns = ['student_count', 'bus_stop_lat', 'bus_stop_lon']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid data with better validation
            df = df.dropna(subset=numeric_columns)
            df = df[df['student_count'] > 0]
            
            # Validate coordinate ranges
            df = df[
                (df['bus_stop_lat'].between(-90, 90)) & 
                (df['bus_stop_lon'].between(-180, 180))
            ]
            
            if len(df) == 0:
                raise ValueError("No valid data points after cleaning")
            
            # Convert to BoardingPoint objects
            boarding_points = []
            for _, row in df.iterrows():
                point = BoardingPoint(
                    cluster_id=str(row['cluster']),
                    student_count=int(row['student_count']),
                    latitude=float(row['bus_stop_lat']),
                    longitude=float(row['bus_stop_lon'])
                )
                boarding_points.append(point)
            
            logger.info(f"✅ Loaded {len(boarding_points)} boarding points")
            return boarding_points
            
        except Exception as e:
            logger.error(f"❌ Error loading boarding points: {e}")
            raise

    def cluster_by_capacity(self, points: List[BoardingPoint]) -> List[List[BoardingPoint]]:
        """Step 1: Cluster boarding points by capacity constraint (55 students per bus)"""
        logger.info(f"🎯 Clustering {len(points)} points with max capacity {MAX_BUS_CAPACITY}")
        
        # Sort points by distance from college (farthest first for better clustering)
        college_coord = (self.college_lat, self.college_lon)
        points_with_distance = [(point, geodesic(college_coord, (point.latitude, point.longitude)).kilometers) 
                               for point in points]
        points_with_distance.sort(key=lambda x: x[1], reverse=True)
        sorted_points = [point for point, _ in points_with_distance]
        
        clusters = []
        current_cluster = []
        current_students = 0
        
        for point in sorted_points:
            if current_students + point.student_count <= MAX_BUS_CAPACITY:
                current_cluster.append(point)
                current_students += point.student_count
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [point]
                current_students = point.student_count
        
        if current_cluster:
            clusters.append(current_cluster)
        
        logger.info(f"✅ Created {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            total_students = sum(p.student_count for p in cluster)
            logger.info(f"  Cluster {i+1}: {len(cluster)} points, {total_students} students")
        
        return clusters

    def check_google_directions_accessibility(self, points: List[BoardingPoint]) -> bool:
        """Step 2: Check if route is accessible using Google Directions API"""
        try:
            if not points:
                return True
            
            # Build waypoints for Google Directions API
            waypoints = []
            for point in points:
                waypoints.append(f"{point.latitude},{point.longitude}")
            
            # Create route: College → Points → College
            origin = f"{self.college_lat},{self.college_lon}"
            destination = f"{self.college_lat},{self.college_lon}"
            waypoints_str = "|".join(waypoints)
            
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                'origin': origin,
                'destination': destination,
                'waypoints': waypoints_str,
                'key': self.google_api_key,
                'mode': 'driving',
                'avoid': 'highways',  # Prefer local roads for bus routes
                'optimize': 'true'  # Let Google optimize the route
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    logger.info(f"✅ Google Directions API: Route accessible")
                    return True
                else:
                    logger.warning(f"⚠️ Google Directions API: {data.get('status')} - {data.get('error_message', 'Unknown error')}")
                    return False
            else:
                logger.warning(f"⚠️ Google Directions API request failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"❌ Error checking Google Directions accessibility: {e}")
            return False

    def get_google_distance_matrix(self, points: List[BoardingPoint]) -> Tuple[List[List[float]], List[List[float]]]:
        """Get distance and duration matrix using Google Distance Matrix API"""
        try:
            # Prepare origins and destinations
            origins = [f"{self.college_lat},{self.college_lon}"]  # College
            destinations = [f"{point.latitude},{point.longitude}" for point in points]  # Boarding points
            
            # Add college as destination too
            destinations.append(f"{self.college_lat},{self.college_lon}")
            
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                'origins': '|'.join(origins),
                'destinations': '|'.join(destinations),
                'key': self.google_api_key,
                'mode': 'driving',
                'avoid': 'highways'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    # Extract distance and duration matrices
                    distance_matrix = []
                    duration_matrix = []
                    
                    for row in data['rows'][0]['elements']:
                        if row['status'] == 'OK':
                            distance_matrix.append(row['distance']['value'] / 1000)  # Convert to km
                            duration_matrix.append(row['duration']['value'] / 3600)  # Convert to hours
                        else:
                            distance_matrix.append(float('inf'))
                            duration_matrix.append(float('inf'))
                    
                    logger.info(f"✅ Google Distance Matrix API: Got {len(distance_matrix)} distances")
                    return [distance_matrix], [duration_matrix]
            
            logger.warning(f"⚠️ Google Distance Matrix API failed: {response.status_code}")
            
        except Exception as e:
            logger.warning(f"❌ Error getting Google Distance Matrix: {e}")
        
        # Fallback: Use geodesic distances
        logger.info("Using fallback geodesic distances")
        n = len(points) + 1  # +1 for college
        distance_matrix = [[0] * n for _ in range(n)]
        duration_matrix = [[0] * n for _ in range(n)]
        
        # College coordinates
        college_coord = (self.college_lat, self.college_lon)
        
        # Fill matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i][j] = 0
                    duration_matrix[i][j] = 0
                else:
                    if i == 0:  # From college
                        if j == n-1:  # To college
                            coord2 = college_coord
                        else:
                            coord2 = (points[j-1].latitude, points[j-1].longitude)
                        coord1 = college_coord
                    elif j == 0:  # To college
                        coord1 = (points[i-1].latitude, points[i-1].longitude)
                        coord2 = college_coord
                    elif j == n-1:  # To college
                        coord1 = (points[i-1].latitude, points[i-1].longitude)
                        coord2 = college_coord
                    else:
                        coord1 = (points[i-1].latitude, points[i-1].longitude)
                        coord2 = (points[j-1].latitude, points[j-1].longitude)
                    
                    distance = geodesic(coord1, coord2).kilometers
                    duration = distance / 30  # Assume 30 km/h average speed
                    
                    distance_matrix[i][j] = distance
                    duration_matrix[i][j] = duration
        
        return distance_matrix, duration_matrix

    def solve_tsp_with_ortools(self, points: List[BoardingPoint]) -> Tuple[List[int], float, float]:
        """Step 3: Solve TSP using OR-Tools with Google Distance Matrix"""
        if not points:
            return [], 0, 0
        
        logger.info(f"🎯 Solving TSP for {len(points)} points using OR-Tools")
        
        # Get distance matrix from Google API
        distance_matrix, duration_matrix = self.get_google_distance_matrix(points)
        
        # Create routing model
        n_locations = len(points) + 1  # +1 for college
        manager = pywrapcp.RoutingIndexManager(n_locations, 1, 0)  # Start from college (index 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[0][to_node] * 1000)  # Convert to meters
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Force the route to end at college
        college_idx = n_locations - 1
        routing.AddDisjunction([manager.NodeToIndex(college_idx)], 0)
        
        # Enhanced search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 30
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extract route
            route_indices = []
            index = routing.Start(0)
            
            while not routing.IsEnd(index):
                route_indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            
            route_indices.append(manager.IndexToNode(index))
            
            # Calculate total distance and duration
            total_distance = 0
            total_duration = 0
            
            for i in range(len(route_indices) - 1):
                from_idx = route_indices[i]
                to_idx = route_indices[i + 1]
                total_distance += distance_matrix[0][to_idx]
                total_duration += duration_matrix[0][to_idx]
            
            logger.info(f"✅ TSP solved: {total_distance:.1f}km, {total_duration:.1f}h")
            return route_indices, total_distance, total_duration
        
        logger.warning("❌ TSP solver failed, using simple ordering")
        # Fallback: simple college → points → college
        fallback_route = [0] + list(range(1, n_locations)) + [0]
        total_distance = sum(distance_matrix[0][i] for i in range(1, n_locations)) * 2
        total_duration = sum(duration_matrix[0][i] for i in range(1, n_locations)) * 2
        
        return fallback_route, total_distance, total_duration

    def get_google_directions_route(self, points: List[BoardingPoint], route_order: List[int]) -> Dict:
        """Get detailed route using Google Directions API"""
        try:
            # Build waypoints in optimized order
            waypoints = []
            for idx in route_order[1:-1]:  # Skip start and end (college)
                if idx > 0 and idx <= len(points):
                    point = points[idx-1]
                    waypoints.append(f"{point.latitude},{point.longitude}")
            
            origin = f"{self.college_lat},{self.college_lon}"
            destination = f"{self.college_lat},{self.college_lon}"
            waypoints_str = "|".join(waypoints)
            
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                'origin': origin,
                'destination': destination,
                'waypoints': waypoints_str,
                'key': self.google_api_key,
                'mode': 'driving',
                'avoid': 'highways',
                'optimize': 'false'  # Don't optimize, use our TSP order
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK' and data.get('routes'):
                    route = data['routes'][0]
                    
                    # Extract polyline
                    polyline_points = []
                    for leg in route['legs']:
                        for step in leg['steps']:
                            if 'polyline' in step and 'points' in step['polyline']:
                                # Decode polyline points
                                decoded_points = polyline.decode(step['polyline']['points'])
                                polyline_points.extend(decoded_points)
                    
                    # Encode complete polyline
                    encoded_polyline = polyline.encode(polyline_points)
                    
                    return {
                        'distance_km': route['legs'][0]['distance']['value'] / 1000,
                        'duration_hours': route['legs'][0]['duration']['value'] / 3600,
                        'polyline': encoded_polyline,
                        'success': True,
                        'route_quality': 'high'
                    }
            
            logger.warning(f"Google Directions API failed: {response.status_code}")
            
        except Exception as e:
            logger.warning(f"Error getting Google Directions route: {e}")
        
        # Fallback calculation
        total_distance = 0
        for i in range(len(route_order) - 1):
            idx1, idx2 = route_order[i], route_order[i + 1]
            
            if idx1 == 0:
                coord1 = (self.college_lat, self.college_lon)
            else:
                coord1 = (points[idx1-1].latitude, points[idx1-1].longitude)
                
            if idx2 == 0:
                coord2 = (self.college_lat, self.college_lon)
            else:
                coord2 = (points[idx2-1].latitude, points[idx2-1].longitude)
            
            total_distance += geodesic(coord1, coord2).kilometers
        
        return {
            'distance_km': total_distance,
            'duration_hours': total_distance / 25,  # 25 km/h average
            'polyline': '',
            'success': False,
            'route_quality': 'estimated'
        }

    def process_all_clusters(self, clusters: List[List[BoardingPoint]]) -> List[OptimizedRoute]:
        """Process all clusters and generate optimized routes"""
        all_routes = []
        bus_counter = 1
        
        for cluster_idx, cluster_points in enumerate(clusters):
            logger.info(f"\n🎯 Processing Cluster {cluster_idx + 1} with {len(cluster_points)} points")
            
            # Step 2: Check Google Directions accessibility
            is_accessible = self.check_google_directions_accessibility(cluster_points)
            
            if not is_accessible:
                logger.warning(f"⚠️ Cluster {cluster_idx + 1} may have accessibility issues")
            
            # Step 3: Solve TSP for optimal route
            route_order, distance, duration = self.solve_tsp_with_ortools(cluster_points)
            
            if not route_order:
                logger.error(f"❌ Failed to generate route for cluster {cluster_idx + 1}")
                continue
            
            # Step 4: Get detailed route with Google Directions
            route_details = self.get_google_directions_route(cluster_points, route_order)
            
            # Create route stops
            stops = []
            
            # Add college as starting point
            stops.append({
                'type': 'college_start',
                'name': 'College (Start)',
                'latitude': self.college_lat,
                'longitude': self.college_lon,
                'students': 0,
                'stop_order': 0
            })
            
            # Add boarding points in optimized order
            total_students = 0
            stop_order = 1
            
            for idx in route_order[1:-1]:  # Skip start and end (college)
                if idx > 0 and idx <= len(cluster_points):
                    point = cluster_points[idx-1]
                    stops.append({
                        'type': 'boarding_point',
                        'name': f'Boarding Point {stop_order}',
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'students': point.student_count,
                        'stop_order': stop_order,
                        'cluster_id': point.cluster_id
                    })
                    total_students += point.student_count
                    stop_order += 1
            
            # Add college as ending point
            stops.append({
                'type': 'college_end',
                'name': 'College (End)',
                'latitude': self.college_lat,
                'longitude': self.college_lon,
                'students': 0,
                'stop_order': stop_order
            })
            
            # Create optimized route
            optimized_route = OptimizedRoute(
                bus_id=bus_counter,
                cluster_id=f"cluster_{cluster_idx + 1}",
                total_students=total_students,
                stops=stops,
                total_distance_km=round(route_details['distance_km'], 2),
                total_duration_hours=round(route_details['duration_hours'], 2),
                polyline=route_details['polyline'],
                accessible=is_accessible
            )
            
            all_routes.append(optimized_route)
            
            logger.info(f"✅ Bus {bus_counter}: {len(stops)-2} stops, {total_students} students, "
                      f"{route_details['distance_km']:.1f}km, {route_details['duration_hours']:.1f}h")
            
            bus_counter += 1
        
        self.optimized_routes = all_routes
        return all_routes

    def get_optimization_summary(self) -> Dict:
        """Get comprehensive optimization summary"""
        if not self.optimized_routes:
            return {'error': 'No routes generated'}
        
        total_students = sum(route.total_students for route in self.optimized_routes)
        total_distance = sum(route.total_distance_km for route in self.optimized_routes)
        total_duration = sum(route.total_duration_hours for route in self.optimized_routes)
        accessible_routes = sum(1 for route in self.optimized_routes if route.accessible)
        
        return {
            'success': True,
            'summary': {
                'total_buses_needed': len(self.optimized_routes),
                'total_students_covered': total_students,
                'total_fleet_distance_km': round(total_distance, 2),
                'total_fleet_duration_hours': round(total_duration, 2),
                'accessible_routes': accessible_routes,
                'inaccessible_routes': len(self.optimized_routes) - accessible_routes,
                'average_students_per_bus': round(total_students / len(self.optimized_routes), 1),
                'average_distance_per_route': round(total_distance / len(self.optimized_routes), 1),
                'capacity_utilization': round((total_students / (len(self.optimized_routes) * MAX_BUS_CAPACITY)) * 100, 1)
            },
            'routes': [
                {
                    'bus_id': route.bus_id,
                    'cluster_id': route.cluster_id,
                    'total_students': route.total_students,
                    'total_stops': len([stop for stop in route.stops if stop['type'] == 'boarding_point']),
                    'distance_km': route.total_distance_km,
                    'duration_hours': route.total_duration_hours,
                    'accessible': route.accessible,
                    'capacity_utilization': round((route.total_students / MAX_BUS_CAPACITY) * 100, 1),
                    'stops': route.stops,
                    'polyline': route.polyline
                }
                for route in self.optimized_routes
            ]
        }

# Flask API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "Google-Based Bus Route Optimizer API is running",
        "college_location": {"lat": COLLEGE_LAT, "lon": COLLEGE_LON},
        "version": "2.0",
        "features": ["Google_Directions_API", "Google_Distance_Matrix_API", "OR_Tools_TSP", "Capacity_Clustering"]
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_routes():
    """Main optimization endpoint using Google APIs"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No CSV file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        # Read CSV data
        try:
            csv_content = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            if df.empty:
                return jsonify({"error": "CSV file is empty"}), 400
                
        except UnicodeDecodeError:
            return jsonify({"error": "Invalid file encoding. Please use UTF-8 encoded CSV"}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400

        # Validate CSV structure
        required_columns = ['cluster', 'student_count', 'bus_stop_lat', 'bus_stop_lon']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                "error": f"CSV missing required columns: {missing_columns}",
                "required_columns": required_columns,
                "found_columns": list(df.columns)
            }), 400

        # Initialize optimizer
        optimizer = GoogleBasedRouteOptimizer(COLLEGE_LAT, COLLEGE_LON, GOOGLE_API_KEY)
        
        start_time = time.time()
        
        # Step 1: Load boarding points
        try:
            boarding_points = optimizer.load_boarding_points(df)
        except ValueError as ve:
            return jsonify({"error": f"Data validation failed: {str(ve)}"}), 400
        
        if not boarding_points:
            return jsonify({"error": "No valid boarding points found in the uploaded file"}), 400
        
        # Step 2: Cluster by capacity
        clusters = optimizer.cluster_by_capacity(boarding_points)
        
        if not clusters:
            return jsonify({"error": "No valid clusters could be created"}), 400
        
        # Step 3: Process all clusters
        optimized_routes = optimizer.process_all_clusters(clusters)
        
        if not optimized_routes:
            return jsonify({"error": "No valid routes could be generated"}), 400
        
        # Get comprehensive summary
        summary = optimizer.get_optimization_summary()
        summary['processing_time_seconds'] = round(time.time() - start_time, 2)
        summary['input_summary'] = {
            'total_boarding_points': len(boarding_points),
            'total_clusters': len(clusters),
            'routes_generated': len(optimized_routes)
        }
        
        logger.info(f"🎉 Generated {len(optimized_routes)} routes in {time.time() - start_time:.2f} seconds")
        
        return jsonify({
            "success": True,
            "message": f"Successfully generated {len(optimized_routes)} optimized routes using Google APIs",
            "data": summary
        })
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({"error": f"Optimization failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Google-Based Bus Route Optimizer API...")
    logger.info(f"📍 College Location: {COLLEGE_LAT}, {COLLEGE_LON}")
    logger.info(f"🚌 Max Capacity: {MAX_BUS_CAPACITY} students")
    logger.info(f"📏 Max Distance: {MAX_DISTANCE_KM} km")
    logger.info(f"⏱️ Max Duration: {MAX_DURATION_HOURS} hours")
    logger.info("✨ Features: Google Directions API, Google Distance Matrix API, OR-Tools TSP")
    app.run(debug=True, host='0.0.0.0', port=5000)

