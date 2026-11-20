import argparse
import sys
import math

def analyze_pos_file(file_path):
    lats = []
    lons = []
    heights = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('%') or not line.strip():
                    continue
                
                parts = line.split()
                # Ensure the line has enough columns
                if len(parts) >= 6:
                    q_flag = parts[5]
                    
                    # Process only FIXED points (Q=1)
                    if q_flag == '1':
                        try:
                            lats.append(float(parts[2]))
                            lons.append(float(parts[3]))
                            heights.append(float(parts[4]))
                        except ValueError:
                            continue

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    if not lats:
        print("No FIXED (Q=1) positions found in the provided file.")
        sys.exit(0)

    # Calculate Min, Max, and Variations
    max_lat, min_lat = max(lats), min(lats)
    max_lon, min_lon = max(lons), min(lons)
    max_hgt, min_hgt = max(heights), min(heights)

    var_lat_deg = max_lat - min_lat
    var_lon_deg = max_lon - min_lon
    var_hgt_m = max_hgt - min_hgt

    # Convert degree variations to metric distances (approximate for local analysis)
    # 1 degree of latitude is approx 111,320 meters
    avg_lat_rad = math.radians(sum(lats) / len(lats))
    var_lat_mm = var_lat_deg * 111320.0 * 1000.0
    var_lon_mm = var_lon_deg * (111320.0 * math.cos(avg_lat_rad)) * 1000.0
    var_hgt_mm = var_hgt_m * 1000.0

    # Print Report
    print("-" * 60)
    print(f"POSITION VARIATION ANALYSIS (FIXED POINTS ONLY)")
    print("-" * 60)
    print(f"Total FIXED epochs analyzed: {len(lats)}")
    print("-" * 60)
    
    print("LATITUDE:")
    print(f"  Max: {max_lat:.9f} deg")
    print(f"  Min: {min_lat:.9f} deg")
    print(f"  Variation: {var_lat_deg:.9f} deg  (~ {var_lat_mm:.2f} mm)")
    print()
    
    print("LONGITUDE:")
    print(f"  Max: {max_lon:.9f} deg")
    print(f"  Min: {min_lon:.9f} deg")
    print(f"  Variation: {var_lon_deg:.9f} deg  (~ {var_lon_mm:.2f} mm)")
    print()
    
    print("HEIGHT:")
    print(f"  Max: {max_hgt:.4f} m")
    print(f"  Min: {min_hgt:.4f} m")
    print(f"  Variation: {var_hgt_m:.4f} m        (~ {var_hgt_mm:.2f} mm)")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze maximum/minimum variations of FIXED positions in a .pos file.")
    parser.add_argument("pos_file", type=str, help="Path to the .pos file")
    
    args = parser.parse_args()
    analyze_pos_file(args.pos_file)