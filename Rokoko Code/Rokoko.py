import socket
import json
import pandas as pd
import time
import warnings
warnings.simplefilter('ignore')

# Unified function to receive live data and store it in a DataFrame
def live_data_to_df(data, column_names, noHeader):
    body_data = []
    
    # Decode the received data
    try:
        d = json.loads(data)
    except json.JSONDecodeError:
        print("Error decoding JSON data")
        return None, column_names, noHeader
    
    # Ensure the data structure exists
    if "scene" not in d or "actors" not in d["scene"] or len(d["scene"]["actors"]) == 0:
        print("No actors or scene data found")
        return None, column_names, noHeader
    
    body = d["scene"]["actors"][0].get("body", {})
    if not body:
        print("No body data available")
        return None, column_names, noHeader

    body_parts = list(body.keys())

    # Dynamically create column headers if not already created
    if noHeader:
        for body_part in body_parts:
            if "right" in body_part:
                if any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
                    another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
                                    body_part + "_rotation_x", body_part + "_rotation_y", body_part + "_rotation_z", body_part + "_rotation_w"]
                    column_names.extend(another_temp)
        df = pd.DataFrame(columns=column_names)
        noHeader = False
    else:
        df = pd.DataFrame(columns=column_names)

    # Extract body part data (position and rotation) and detect movements
    for body_part in body_parts:
        if "right" in body_part:
            if any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
                b = body.get(body_part, {})
                if "position" in b and "rotation" in b:
                    # Check movement direction based on Y and Z position values
                    if b["position"]["x"] < 0:
                        print("Glove moved to the left")
                    elif b["position"]["x"] > 0:
                        print("Glove moved to the right")
                    if b["position"]["y"] > 0:
                        print("Glove moved up")
                    elif b["position"]["y"] < 0:
                        print("Glove moved down")

                    # Append position and rotation data
                    temp = [b["position"]["x"], b["position"]["y"], b["position"]["z"], 
                            b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"], b["rotation"]["w"]]
                    body_data.extend(temp)
    
    if body_data:
        df.loc[len(df)] = body_data
    else:
        return None, column_names, noHeader

    return df, column_names, noHeader

# Function to set a preset starting position
def set_preset_position():
    # Define a preset position (example values, adjust as necessary)
    preset_position = [0.0, 0.0, 0.0,   # positionX, positionY, positionZ
                       0.0, 0.0, 0.0, 1.0]  # rotation_x, rotation_y, rotation_z, rotation_w

    # Create a DataFrame with the correct columns
    columns = ["positionX", "positionY", "positionZ", "rotation_x", "rotation_y", "rotation_z", "rotation_w"]
    df = pd.DataFrame(columns=columns)

    # Add the preset position to the DataFrame
    df.loc[len(df)] = preset_position
    return df

# Main loop to receive data and process it
def main():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 14043

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Step 1: Set and save the preset position
    print("Setting preset position...")
    df = set_preset_position()  # Set the initial position
    df.to_csv("preset_data.csv", index=False)  # Save preset data
    print("Preset data recorded in 'preset_data.csv'.")

    # Step 2: Wait for user input to continue
    input("Press Enter to start continuous data recording...")

    packet_count = 0
    start_time = time.time()
    duration = 10  # Monitor the frequency for a set period (e.g., 10 seconds)

    print("Press Ctrl+C to stop data recording.")

    try:
        while True:
            # Receive data from the socket
            data, addr = sock.recvfrom(65000)
            data = data.decode("utf-8")
            
            # Process the data using live_data_to_df
            live_df_data, column_names, noHeader = live_data_to_df(data, [], True)
            
            if live_df_data is not None:
                # Increase packet count for each successful data reception
                packet_count += 1

                # Save data to a CSV file continuously
                live_df_data.to_csv("live_data.csv", mode='a', header=False, index=False)

            # Calculate and print the average recording frequency every set period
            if time.time() - start_time >= duration:
                avg_frequency = packet_count / duration
                print(f"Average recording frequency over the last {duration} seconds: {avg_frequency:.2f} packets/second")
                
                # Reset for the next interval
                packet_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("Data recording stopped.")
        print("Final data saved to live_data.csv.")

if __name__ == "__main__":
    main()