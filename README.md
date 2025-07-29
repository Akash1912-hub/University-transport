# Rajalakshmi Transport Boarding Points System

A comprehensive transport management system that visualizes boarding points and optimizes bus routes using advanced algorithms.

## Features

- 📁 **CSV File Upload**: Upload CSV files with boarding point data
- 🗺️ **Interactive Map**: Visualize boarding points on Google Maps
- 🚌 **Bus Icons**: Custom bus icons for boarding points
- 🎯 **Route Optimization**: Advanced routing algorithm using OR-Tools
- 📊 **Statistics Dashboard**: View total points, students, and routes
- 💬 **Info Windows**: Click on markers to see detailed information
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎓 **Educational Theme**: Professional design for educational institutions

## System Architecture

### Frontend
- **HTML/CSS/JavaScript**: Modern, responsive web interface
- **Google Maps API**: Interactive map visualization
- **Font Awesome**: Professional icons throughout the interface

### Backend
- **Flask**: Python web server
- **OR-Tools**: Google's optimization library for route planning
- **OSMnx**: OpenStreetMap data processing
- **NetworkX**: Graph algorithms for routing
- **Pandas**: Data processing and CSV handling

## Required CSV Format

Your CSV file must contain the following columns:
- `cluster`: The cluster/group identifier
- `student_count`: Number of students at this boarding point
- `bus_stop_lat`: Latitude of the bus stop
- `bus_stop_lon`: Longitude of the bus stop

### Example CSV Format:
```csv
cluster,student_count,bus_stop_lat,bus_stop_lon
Cluster_A,25,40.7128,-74.0060
Cluster_B,32,40.7505,-73.9934
Cluster_C,28,40.7614,-73.9776
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Node.js (optional, for development)

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask server**:
   ```bash
   python app.py
   ```
   The server will run on `http://localhost:5000`

### Frontend Setup

1. **Open the application**:
   - Simply open `index.html` in your web browser
   - Or serve it using a local server

2. **Access the application**:
   - Open `http://localhost:5000` (if using a server)
   - Or directly open `index.html` in your browser

## How to Use

### Step 1: Upload Data
1. Click "Choose CSV File" and select your CSV file
2. Click "Display Points" to visualize the boarding points on the map
3. View the statistics and cluster information

### Step 2: Optimize Routes
1. Click "Optimize Routes" to run the routing algorithm
2. Wait for the optimization to complete (may take a few minutes)
3. View the optimized routes with different colored bus icons
4. Check the route summary statistics

### Features Explained

#### Map Visualization
- **Bus Icons**: Each boarding point is represented by a colored bus icon
- **College Icon**: Special purple icon for the college location
- **Color-coded Routes**: Different colors for different transport routes
- **Info Windows**: Click any marker to see detailed information

#### Route Optimization
- **OR-Tools Algorithm**: Advanced vehicle routing optimization
- **Capacity Constraints**: Maximum 55 students per bus
- **Distance Limits**: Maximum 40km per route
- **Time Constraints**: Maximum 120 minutes per route
- **College Centered**: All routes start and end at the college

#### Statistics Dashboard
- **Total Boarding Points**: Number of valid boarding points loaded
- **Total Students**: Sum of all students across all boarding points
- **Total Routes**: Number of optimized transport routes
- **Route Summary**: Distance, time, and student statistics

## API Endpoints

### POST `/api/route`
Upload a CSV file to run the routing algorithm.

**Request**: Multipart form data with CSV file
**Response**: JSON with optimized routes and statistics

### GET `/api/health`
Health check endpoint.

**Response**: `{"status": "healthy"}`

## Algorithm Details

The routing algorithm uses:
- **OSMnx**: Fetches real road network data from OpenStreetMap
- **NetworkX**: Computes shortest paths between locations
- **OR-Tools**: Solves the Vehicle Routing Problem (VRP)
- **Constraints**: Bus capacity, route distance, and time limits

## Configuration

You can modify the algorithm parameters in `app.py`:

```python
COLLEGE_LAT = 13.0827          # College latitude
COLLEGE_LON = 80.2707          # College longitude
MAX_STUDENTS_PER_BUS = 55      # Maximum students per bus
MAX_ROUTE_DISTANCE_KM = 40     # Maximum route distance
MAX_ROUTE_TIME_MIN = 120       # Maximum route time
NUM_BUSES = 100                # Number of available buses
```

## Browser Compatibility

Works on all modern browsers:
- Chrome
- Firefox
- Safari
- Edge

## Sample Data

A sample CSV file (`sample_data.csv`) is included with the application for testing purposes.

## API Key

The application uses the Google Maps API key: `AIzaSyBlfqs5K9HEe9c1Eu5bjPXXjr8Hz2mbTZE`

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure the Flask server is running and CORS is enabled
2. **File Upload Issues**: Ensure the CSV file has the correct format
3. **Routing Errors**: Check that all coordinates are valid and accessible
4. **Performance**: Large datasets may take several minutes to process

### Error Messages

- **"Missing required columns"**: Check your CSV format
- **"No accessible boarding points"**: Coordinates may be outside the road network
- **"Algorithm error"**: Check the server logs for detailed error information

## Development

### Project Structure
```
├── index.html          # Frontend application
├── app.py             # Flask backend server
├── requirements.txt    # Python dependencies
├── sample_data.csv    # Sample data for testing
└── README.md         # This file
```

### Adding Features
- **New Map Styles**: Modify the map styles array in `index.html`
- **Additional Constraints**: Update the routing algorithm in `app.py`
- **UI Improvements**: Modify CSS styles in `index.html`

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the project repository. 