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

    # Extract body part data (position and rotation)
    for body_part in body_parts:
        if "right" in body_part:
            if any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
                b = body.get(body_part, {})
                if "position" in b and "rotation" in b:
                    temp = [b["position"]["x"], b["position"]["y"], b["position"]["z"], 
                            b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"], b["rotation"]["w"]]
                    body_data.extend(temp)
    
    if body_data:
        df.loc[len(df)] = body_data
    else:
        return None, column_names, noHeader

    return df, column_names, noHeader

# Main loop to receive data and process it
def main():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 14043

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    continue_gesture = True
    column_names = []
    noHeader = True

    while continue_gesture:
        choice = input("Do you want to add a new gesture? (yes or no).\n")
        if choice == 'yes':
            gesture_name = input("Okay. What is the name of the gesture?\n")
            trials = int(input("Okay. How many trials will you perform?\n"))
            duration = int(input("Okay. How long do you want perform the gesture for? (seconds)\n"))

            for trial in range(trials):
                print(f"Preparing for trial {trial + 1} for gesture '{gesture_name}'...")
                input("Press enter to start.")
                
                print("Recording data...")
                start_time = time.time()
                dfs = []  # Store data for this trial
                while time.time() - start_time < duration:
                    # Receive data from the socket
                    data, addr = sock.recvfrom(65000)
                    data = data.decode("utf-8")
                    
                    # Process the data using live_data_to_df
                    live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
                    
                    if live_df_data is not None:
                        dfs.append(live_df_data)
                    else:
                        print("Failed to capture data for this segment.")

                if dfs:
                    trial_df = pd.concat(dfs, ignore_index=True)
                    trial_df["Gesture"] = gesture_name
                    
                    # Save this trial's data to a CSV file
                    filename = f"{gesture_name}_trial_{trial + 1}.csv"
                    trial_df.to_csv(filename, index=False)
                    print(f"Trial {trial + 1} data saved to {filename}.")
                else:
                    print(f"No data captured for trial {trial + 1}.")

            print(f"Completed {trials} trials for gesture '{gesture_name}'.")
        
        elif choice == 'no':
            continue_gesture = False
            print("Exiting program. No more gestures to capture.")

if __name__ == "__main__":
    main()
