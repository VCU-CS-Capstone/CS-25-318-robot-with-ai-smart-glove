import socket
import json
import pandas as pd
import time
import os
import warnings
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.simplefilter('ignore')

previous_data = {}  # store previous sensor data for movement comparison

# Function to parse incoming live data, process it, and store in a DataFrame
def live_data_to_df(data, column_names, noHeader):
    body_data = []  # List to store parsed data for each body part
    # Attempt to decode received JSON data
    try:
        d = json.loads(data)
    except json.JSONDecodeError:
        print("Error decoding JSON data")
        return None, column_names, noHeader
    
    if "scene" not in d or "actors" not in d["scene"] or len(d["scene"]["actors"]) == 0:
        print("No actors or scene data found")
        return None, column_names, noHeader
    
    body = d["scene"]["actors"][0].get("body", {})
    if not body:
        print("No body data available")
        return None, column_names, noHeader

    body_parts = list(body.keys())


    # Add header if it's the first run
    if noHeader:
        for body_part in body_parts:
            if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
                another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
                                body_part + "_rotation_x", body_part + "_rotation_y", body_part + "_rotation_z", body_part + "_rotation_w"]
                column_names.extend(another_temp)
        column_names.extend(["timestamp"])  # Add direction and timestamp columns
        noHeader = False

    left_index_distal_position_x = None
    left_index_proximal_position_x = None
    # left_middle_distal_position_x = None
    # left_middle_proximal_position_x = None
    # left_ring_distal_position_x = None
    # left_ring_proximal_position_x = None
    # left_little_distal_position_x = None
    # left_little_proximal_position_x = None

    # left_index_distal_position_y = None
    # left_index_proximal_position_y = None
    # left_middle_distal_position_y = None
    # left_middle_proximal_position_y = None
    # left_ring_distal_position_y = None
    # left_ring_proximal_position_y = None
    # left_little_distal_position_y = None
    # left_little_proximal_position_y = None

    # left_index_distal_position_z = None
    # left_index_proximal_position_z = None
    # left_middle_distal_position_z = None
    # left_middle_proximal_position_z = None
    # left_ring_distal_position_z = None
    # left_ring_proximal_position_z = None
    # left_little_distal_position_z = None
    # left_little_proximal_position_z = None

    for body_part in body_parts:
        if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
            b = body.get(body_part, {})
            if "position" in b and "rotation" in b:
                temp = [b["position"]["x"], b["position"]["y"], b["position"]["z"],
                        b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"], b["rotation"]["w"]]
                body_data.extend(temp)

                # Capture the positionX values for leftIndexDistal and leftIndexProximal
                if body_part == "leftIndexDistal":
                    left_index_distal_position_x = b["position"]["x"]
                if body_part == "leftIndexProximal":
                    left_index_proximal_position_x = b["position"]["x"]

                # Calculate the difference between leftIndexDistal_positionX and leftIndexProximal_positionX if both are available
                if left_index_distal_position_x is not None and left_index_proximal_position_x is not None:
                    position_difference = (left_index_distal_position_x - left_index_proximal_position_x)
                    # Check if the difference is between 0.07 and 0.08
                    # print(position_difference)
                    # if 0.06 <= position_difference <= 0.08:
                    #     #print("right")  # Print "right" if the difference is in the specified range
                    #     print(position_difference)
                    # if -0.08 <= position_difference *-1 <= -0.06:
                    #     #print("left")  # Print "right" if the difference is in the specified range
                    #     print(position_difference)

    if body_data:
        body_data.append(time.strftime("%Y-%m-%d %H:%M:%S"))  # Append the timestamp to the data row
        df = pd.DataFrame([body_data], columns=column_names)
        return df, column_names, noHeader
    else:
        return None, column_names, noHeader
    


def set_preset_position():
    # Define a preset position
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
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))


    # Step #1: Set and save the preset position
    print("Setting preset position...")
    df = set_preset_position()  # Set the initial position
    df.to_csv("preset_data.csv", index=False)
    print("Preset data recorded in 'preset_data.csv'.")

    column_names = []
    noHeader = True
    first_write = True

    # Step 2: Wait for user input to continue
    input("Press Enter to start continuous data recording...")

    packet_count = 0
    start_time = time.time()
    duration = 10  # Monitor the frequency for a set period (e.g., 10 seconds)
    # print("Columns in DataFrame:", df.columns)
    # print("Body parts in DataFrame:", df)

    try:
        while True:
            # Start 5-second recording period
            # Receive data from the socket
            print("Recording data for 5 seconds...")
            data, addr = sock.recvfrom(65000)
            data = data.decode("utf-8")
            
            # Process the data using live_data_to_df
            live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)

            if live_df_data is not None:
                # Increase packet count for each successful data reception
                packet_count += 1

                # Save data to a CSV file continuously
                live_df_data.to_csv("live_data.csv", mode='a', header=first_write, index=False)
                first_write = False

            if time.time() - start_time >= duration:
                avg_frequency = packet_count / duration
                print(f"Average recording frequency over the last {duration} seconds: {avg_frequency:.2f} packets/second")
                
                # Reset for the next interval
                packet_count = 0
                start_time = time.time()

            # 5-second waiting period
            print("Waiting for 5 seconds...")
            #time.sleep(5)

    except KeyboardInterrupt:
        print("Data recording stopped.")
        print("Final data saved to live_data.csv.")

if __name__ == "__main__":
    main()