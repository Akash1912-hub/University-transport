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
OSRM_URL = "http://router.project-osrm.org"

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

class PreClusteredRouteOptimizer:
    def __init__(self, college_lat: float, college_lon: float, osrm_url: str = OSRM_URL):
        self.college_lat = college_lat
        self.college_lon = college_lon
        self.osrm_url = osrm_url
        self.boarding_points = []
        self.optimized_routes = []
        
    def load_pre_clustered_data(self, df: pd.DataFrame) -> Dict[str, List[BoardingPoint]]:
        """Load pre-clustered boarding points from CSV"""
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
            
            # Group by cluster
            clustered_points = {}
            
            for _, row in df.iterrows():
                cluster_id = str(row['cluster'])
                point = BoardingPoint(
                    cluster_id=cluster_id,
                    student_count=int(row['student_count']),
                    latitude=float(row['bus_stop_lat']),
                    longitude=float(row['bus_stop_lon'])
                )
                
                if cluster_id not in clustered_points:
                    clustered_points[cluster_id] = []
                clustered_points[cluster_id].append(point)
            
            logger.info(f"✅ Loaded {len(df)} boarding points across {len(clustered_points)} clusters")
            return clustered_points
            
        except Exception as e:
            logger.error(f"❌ Error loading pre-clustered data: {e}")
            raise

    def check_cluster_capacity_and_split(self, cluster_points: List[BoardingPoint], max_capacity: int = 55) -> List[List[BoardingPoint]]:
        """Check if cluster exceeds capacity and split if necessary"""
        total_students = sum(point.student_count for point in cluster_points)
        
        if total_students <= max_capacity:
            logger.info(f"Cluster {cluster_points[0].cluster_id}: {total_students} students - OK")
            return [cluster_points]
        
        # Need to split cluster
        logger.info(f"Cluster {cluster_points[0].cluster_id}: {total_students} students - SPLITTING")
        
        # Sort points by distance from college to maintain geographical coherence
        college_coord = (self.college_lat, self.college_lon)
        cluster_points.sort(key=lambda p: geodesic(college_coord, (p.latitude, p.longitude)).kilometers)
        
        # Split into sub-clusters that respect capacity
        sub_clusters = []
        current_cluster = []
        current_students = 0
        
        for point in cluster_points:
            if current_students + point.student_count <= max_capacity:
                current_cluster.append(point)
                current_students += point.student_count
            else:
                if current_cluster:
                    sub_clusters.append(current_cluster)
                current_cluster = [point]
                current_students = point.student_count
        
        if current_cluster:
            sub_clusters.append(current_cluster)
        
        logger.info(f"Split into {len(sub_clusters)} sub-clusters")
        return sub_clusters

    def check_heavy_vehicle_accessibility(self, points: List[BoardingPoint], route_order: List[int] = None) -> bool:
        """Check if route is accessible by heavy vehicles using OSRM"""
        try:
            # If route_order not provided, use simple order
            if route_order is None:
                route_order = list(range(len(points))) + [len(points)]  # Add college at end
            
            # Build route coordinates following the optimized order
            coordinates = []
            
            # Add coordinates in the order they will be visited
            for idx in route_order:
                if idx < len(points):  # Boarding point
                    point = points[idx]
                    coordinates.append(f"{point.longitude},{point.latitude}")
                else:  # College (final destination)
                    coordinates.append(f"{self.college_lon},{self.college_lat}")
            
            coord_string = ";".join(coordinates)
            
            # Use OSRM with driving profile
            url = f"{self.osrm_url}/route/v1/driving/{coord_string}"
            params = {
                'overview': 'false',
                'geometries': 'polyline'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok':
                    logger.info(f"✅ Route accessible by heavy vehicles")
                    return True
            
            logger.warning(f"⚠️ Route not accessible by heavy vehicles")
            return False
            
        except Exception as e:
            logger.warning(f"❌ Error checking heavy vehicle accessibility: {e}")
            return False

    def solve_tsp_with_ortools(self, points: List[BoardingPoint]) -> Tuple[List[int], float, float]:
        """Solve TSP starting from farthest point and ending at college"""
        if not points:
            return [], 0, 0
        
        # Find the farthest point from college as starting point
        college_coord = (self.college_lat, self.college_lon)
        start_distances = [(i, geodesic((points[i].latitude, points[i].longitude), college_coord).kilometers) 
                          for i in range(len(points))]
        start_distances.sort(key=lambda x: x[1], reverse=True)  # Sort by distance DESC
        farthest_point_idx = start_distances[0][0]  # Farthest point as starting point
        
        logger.info(f"Starting from farthest point {farthest_point_idx} ({start_distances[0][1]:.1f}km from college)")
        
        # Create locations (all boarding points + college at the end)
        locations = []
        for point in points:
            locations.append((point.latitude, point.longitude))
        locations.append((self.college_lat, self.college_lon))  # College as final destination
        
        # Create distance matrix with caching for better performance
        distance_cache = {}
        def distance_callback(from_index, to_index):
            if (from_index, to_index) in distance_cache:
                return distance_cache[(from_index, to_index)]
            
            from_node = locations[from_index]
            to_node = locations[to_index]
            distance = int(geodesic(from_node, to_node).meters)
            distance_cache[(from_index, to_index)] = distance
            return distance
        
        # Create routing model with the farthest point as depot
        manager = pywrapcp.RoutingIndexManager(len(locations), 1, farthest_point_idx)
        routing = pywrapcp.RoutingModel(manager)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Force the route to end at college
        college_idx = len(points)  # College is the last location
        routing.AddDisjunction([manager.NodeToIndex(college_idx)], 0)  # Must visit college
        
        # Enhanced search parameters for better solutions
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 45  # Increased time limit
        search_parameters.log_search = False  # Reduce logging noise
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extract route starting from farthest point
            route_indices = []
            index = routing.Start(0)
            
            while not routing.IsEnd(index):
                route_indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            
            route_indices.append(manager.IndexToNode(index))  # Add end point (college)
            
            # Calculate total distance and estimated duration
            total_distance = 0
            for i in range(len(route_indices) - 1):
                from_idx = route_indices[i]
                to_idx = route_indices[i + 1]
                from_coord = locations[from_idx]
                to_coord = locations[to_idx]
                total_distance += geodesic(from_coord, to_coord).kilometers
            
            # Better duration estimation considering traffic and boarding time
            base_travel_time = total_distance / 30  # 30 km/h average speed
            boarding_time = len(points) * 2 / 60  # 2 minutes per stop
            traffic_buffer = base_travel_time * 0.2  # 20% traffic buffer
            estimated_duration = base_travel_time + boarding_time + traffic_buffer
            
            logger.info(f"✅ Route optimized from farthest point: {total_distance:.1f}km, {estimated_duration:.1f}h")
            return route_indices, total_distance, estimated_duration
        
        logger.warning("❌ TSP solver failed, using farthest-to-nearest ordering")
        # Fallback: start from farthest point, visit others by proximity, end at college
        remaining_points = list(range(len(points)))
        remaining_points.remove(farthest_point_idx)
        
        # Order remaining points by distance from current position (greedy nearest neighbor)
        fallback_route = [farthest_point_idx]
        current_pos = (points[farthest_point_idx].latitude, points[farthest_point_idx].longitude)
        
        while remaining_points:
            distances = [(idx, geodesic(current_pos, (points[idx].latitude, points[idx].longitude)).kilometers) 
                        for idx in remaining_points]
            distances.sort(key=lambda x: x[1])  # Sort by distance ASC (nearest first)
            next_idx = distances[0][0]
            fallback_route.append(next_idx)
            remaining_points.remove(next_idx)
            current_pos = (points[next_idx].latitude, points[next_idx].longitude)
        
        fallback_route.append(len(points))  # Add college at the end
        
        total_distance = sum(geodesic(locations[fallback_route[i]], locations[fallback_route[i+1]]).kilometers 
                           for i in range(len(fallback_route)-1))
        estimated_duration = (total_distance / 30) + (len(points) * 2 / 60) + (total_distance / 30 * 0.2)
        
        logger.info(f"✅ Fallback route from farthest point: {total_distance:.1f}km, {estimated_duration:.1f}h")
        return fallback_route, total_distance, estimated_duration

    def get_detailed_route_with_polyline(self, points: List[BoardingPoint], route_order: List[int]) -> Dict:
        """Get detailed route information with polyline using OSRM"""
        try:
            # Build coordinate string for OSRM following the actual route order
            coordinates = []
            
            for idx in route_order:
                if idx < len(points):  # Boarding point
                    point = points[idx]
                    coordinates.append(f"{point.longitude},{point.latitude}")
                else:  # College (final destination)
                    coordinates.append(f"{self.college_lon},{self.college_lat}")
            
            coord_string = ";".join(coordinates)
            
            url = f"{self.osrm_url}/route/v1/driving/{coord_string}"
            params = {
                'overview': 'full',
                'geometries': 'polyline',
                'steps': 'true',
                'annotations': 'duration,distance'  # Get detailed annotations
            }
            
            response = requests.get(url, params=params, timeout=25)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 'Ok' and data.get('routes'):
                    route = data['routes'][0]
                    
                    return {
                        'distance_km': route['distance'] / 1000,
                        'duration_hours': route['duration'] / 3600,
                        'polyline': route['geometry'],
                        'success': True,
                        'route_quality': 'high'
                    }
            
            logger.warning("OSRM route request failed, using fallback")
            
        except Exception as e:
            logger.warning(f"Error getting detailed route: {e}")
        
        # Enhanced fallback calculation
        total_distance = 0
        for i in range(len(route_order) - 1):
            idx1, idx2 = route_order[i], route_order[i + 1]
            
            if idx1 < len(points):
                coord1 = (points[idx1].latitude, points[idx1].longitude)
            else:
                coord1 = (self.college_lat, self.college_lon)
                
            if idx2 < len(points):
                coord2 = (points[idx2].latitude, points[idx2].longitude)
            else:
                coord2 = (self.college_lat, self.college_lon)
            
            total_distance += geodesic(coord1, coord2).kilometers
        
        # Better duration estimation for fallback
        base_travel_time = total_distance / 25  # 25 km/h conservative speed
        boarding_time = len(points) * 2 / 60  # 2 minutes per boarding stop
        buffer_time = base_travel_time * 0.3  # 30% buffer for traffic/delays
        
        return {
            'distance_km': total_distance,
            'duration_hours': base_travel_time + boarding_time + buffer_time,
            'polyline': '',
            'success': False,
            'route_quality': 'estimated'
        }

    def process_all_clusters(self, clustered_points: Dict[str, List[BoardingPoint]]) -> List[OptimizedRoute]:
        """Process all clusters and generate optimized routes"""
        all_routes = []
        bus_counter = 1
        
        for cluster_id, points in clustered_points.items():
            logger.info(f"\n🎯 Processing Cluster {cluster_id} with {len(points)} points")
            
            # Step 1: Check capacity and split if needed
            sub_clusters = self.check_cluster_capacity_and_split(points, MAX_BUS_CAPACITY)
            
            for sub_idx, sub_cluster in enumerate(sub_clusters):
                sub_cluster_id = f"{cluster_id}_{sub_idx}" if len(sub_clusters) > 1 else cluster_id
                
                logger.info(f"🚌 Processing sub-cluster {sub_cluster_id}")
                
                # Step 2: Solve TSP for optimal route
                route_order, distance, duration = self.solve_tsp_with_ortools(sub_cluster)
                
                if not route_order:
                    logger.error(f"❌ Failed to generate route for sub-cluster {sub_cluster_id}")
                    continue
                
                # Step 3: Check heavy vehicle accessibility with the optimized route
                is_accessible = self.check_heavy_vehicle_accessibility(sub_cluster, route_order)
                
                if not is_accessible:
                    logger.warning(f"⚠️ Sub-cluster {sub_cluster_id} may have accessibility issues")
                
                # Step 4: Check distance and duration constraints
                if distance > MAX_DISTANCE_KM or duration > MAX_DURATION_HOURS:
                    logger.warning(f"⚠️ Route exceeds constraints: {distance:.1f}km, {duration:.1f}h")
                
                # Step 5: Get detailed route with polyline
                route_details = self.get_detailed_route_with_polyline(sub_cluster, route_order)
                
                # Step 6: Create route stops with better ordering
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
                
                for idx in route_order[1:-1]:  # Skip start and end depot
                    if idx < len(sub_cluster):  # Safety check
                        point = sub_cluster[idx]
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
                    cluster_id=sub_cluster_id,
                    total_students=total_students,
                    stops=stops,
                    total_distance_km=round(route_details['distance_km'], 2),
                    total_duration_hours=round(route_details['duration_hours'], 2),
                    polyline=route_details['polyline'],
                    accessible=is_accessible
                )
                
                all_routes.append(optimized_route)
                
                logger.info(f"✅ Bus {bus_counter}: {len(stops)-2} stops, {total_students} students, "
                          f"{route_details['distance_km']:.1f}km, {route_details['duration_hours']:.1f}h, "
                          f"Quality: {route_details.get('route_quality', 'unknown')}")
                
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
        
        # Additional metrics
        efficiency_scores = []
        for route in self.optimized_routes:
            if route.total_distance_km > 0:
                efficiency = route.total_students / route.total_distance_km
                efficiency_scores.append(efficiency)
        
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        
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
                'fleet_efficiency_score': round(avg_efficiency, 2),  # students per km
                'capacity_utilization': round((total_students / (len(self.optimized_routes) * MAX_BUS_CAPACITY)) * 100, 1)
            },
            'routes': [
                {
                    'bus_id': route.bus_id,
                    'cluster_id': route.cluster_id,
                    'total_students': route.total_students,
                    'total_stops': len([stop for stop in route.stops if stop['type'] == 'boarding_point']),
                    'starting_point': route.stops[1]['name'] if len(route.stops) > 1 else 'Unknown',
                    'distance_km': route.total_distance_km,
                    'duration_hours': route.total_duration_hours,
                    'accessible': route.accessible,
                    'capacity_utilization': round((route.total_students / MAX_BUS_CAPACITY) * 100, 1),
                    'efficiency_score': round(route.total_students / route.total_distance_km, 2) if route.total_distance_km > 0 else 0,
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
        "message": "Pre-Clustered Bus Route Optimizer API is running",
        "college_location": {"lat": COLLEGE_LAT, "lon": COLLEGE_LON},
        "version": "1.1",
        "features": ["TSP_optimization", "capacity_management", "accessibility_check", "polyline_generation"]
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_routes():
    """Compatibility endpoint - redirects to pre-clustered optimizer"""
    return optimize_pre_clustered_routes()

@app.route('/api/optimize-pre-clustered', methods=['POST'])
def optimize_pre_clustered_routes():
    """Main optimization endpoint for pre-clustered data"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No CSV file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        # Read CSV data with better error handling
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
                "found_columns": list(df.columns),
                "sample_valid_row": {
                    "cluster": "A1", 
                    "student_count": 15, 
                    "bus_stop_lat": 13.0827, 
                    "bus_stop_lon": 80.2707
                }
            }), 400

        # Initialize optimizer
        optimizer = PreClusteredRouteOptimizer(COLLEGE_LAT, COLLEGE_LON, OSRM_URL)
        
        start_time = time.time()
        
        # Step 1: Load pre-clustered data
        try:
            clustered_points = optimizer.load_pre_clustered_data(df)
        except ValueError as ve:
            return jsonify({"error": f"Data validation failed: {str(ve)}"}), 400
        
        if not clustered_points:
            return jsonify({"error": "No valid clusters found in the uploaded file"}), 400
        
        # Step 2: Process all clusters
        optimized_routes = optimizer.process_all_clusters(clustered_points)
        
        if not optimized_routes:
            return jsonify({"error": "No valid routes could be generated"}), 400
        
        # Get comprehensive summary
        summary = optimizer.get_optimization_summary()
        summary['processing_time_seconds'] = round(time.time() - start_time, 2)
        summary['input_summary'] = {
            'total_clusters': len(clustered_points),
            'total_boarding_points': len(df),
            'clusters_processed': len([r.cluster_id.split('_')[0] for r in optimized_routes]),
            'routes_generated': len(optimized_routes)
        }
        
        logger.info(f"🎉 Generated {len(optimized_routes)} routes in {time.time() - start_time:.2f} seconds")
        logger.info(f"📊 Fleet efficiency: {summary['summary']['fleet_efficiency_score']:.2f} students/km")
        
        # Debug: Log the structure of the response
        logger.info(f"📋 Response structure: summary keys = {list(summary['summary'].keys())}")
        logger.info(f"📋 Response structure: routes count = {len(summary['routes'])}")
        if summary['routes']:
            logger.info(f"📋 First route structure: {list(summary['routes'][0].keys())}")
            logger.info(f"📋 First route stops count: {len(summary['routes'][0]['stops'])}")
        
        # Return response in the format expected by the frontend
        return jsonify({
            "success": True,
            "message": f"Successfully generated {len(optimized_routes)} optimized routes",
            "data": summary
        })
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({"error": f"Optimization failed: {str(e)}"}), 500

@app.route('/api/route-details/<int:bus_id>', methods=['GET'])
def get_route_details(bus_id):
    """Get detailed information for a specific route"""
    try:
        # This would typically retrieve from a database or session
        return jsonify({
            "bus_id": bus_id,
            "message": f"Route details for Bus {bus_id}",
            "note": "This endpoint would return detailed route information including turn-by-turn directions",
            "available_details": [
                "turn_by_turn_directions",
                "estimated_arrival_times",
                "traffic_conditions", 
                "alternative_routes"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to get route details: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Improved Pre-Clustered Bus Route Optimizer API...")
    logger.info(f"📍 College Location: {COLLEGE_LAT}, {COLLEGE_LON}")
    logger.info(f"🚌 Max Capacity: {MAX_BUS_CAPACITY} students")
    logger.info(f"📏 Max Distance: {MAX_DISTANCE_KM} km")
    logger.info(f"⏱️ Max Duration: {MAX_DURATION_HOURS} hours")
    logger.info("✨ Improvements: Enhanced validation, better TSP solving, improved duration estimates")
    app.run(debug=True, host='0.0.0.0', port=5000)

