import socket
import json
import pandas as pd
import time
import os
import warnings
warnings.simplefilter('ignore')

previous_data = {}  # store previous sensor data for movement comparison

# Function to determine direction and return a label for CSV output
def determine_direction(diff_x, diff_y):
    if diff_y > 0:
        return "up"
    elif diff_y < 0:
        return "down"
    elif diff_x > 0:
        return "right"
    elif diff_x < 0:
        return "left"
    return None

# Function to parse incoming live data, process it, and store in a DataFrame
def live_data_to_df(data, column_names, noHeader):
    body_data = []  # List to store parsed data for each body part
    direction_data = None  # Variable to store direction for CSV output

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
        column_names.extend(["direction", "timestamp"])  # Add direction and timestamp columns
        noHeader = False

    for body_part in body_parts:
        if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
            b = body.get(body_part, {})
            if "position" in b and "rotation" in b:
                if "Distal" in body_part and "Proximal" in body_part:
                    finger_part = ''.join([ch for ch in body_part if ch.isalpha()])
                    distal_part = finger_part + "Distal"
                    proximal_part = finger_part + "Proximal"

                    if distal_part in body and proximal_part in body:
                        distal_pos = body[distal_part]["position"]
                        proximal_pos = body[proximal_part]["position"]

                        if distal_pos and proximal_pos:
                            diff_x = distal_pos["x"] - proximal_pos["x"]
                            diff_y = distal_pos["y"] - proximal_pos["y"]

                            # Determine direction based on x and y values
                            direction_data = determine_direction(diff_x, diff_y)

                temp = [b["position"]["x"], b["position"]["y"], b["position"]["z"],
                        b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"], b["rotation"]["w"]]
                body_data.extend(temp)

    if body_data:
        body_data.append(direction_data)  # Append the direction to the data row
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

    print("Setting preset position...")
    df = set_preset_position()  # Set the initial position
    df.to_csv("preset_data.csv", index=False)
    print("Preset data recorded in 'preset_data.csv'.")

    column_names = []
    noHeader = True
    first_write = True

    input("Press Enter to start continuous data recording...")

    try:
        while True:
            # Start 5-second recording period
            print("Recording data for 5 seconds...")
            data, addr = sock.recvfrom(65000)
            data = data.decode("utf-8")
            
            live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
            
            if live_df_data is not None:
                live_df_data.to_csv("live_data.csv", mode='a', header=first_write, index=False)
                first_write = False

            # 5-second waiting period
            print("Waiting for 5 seconds...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("Data recording stopped.")
        print("Final data saved to live_data.csv.")

if __name__ == "__main__":
    main()
