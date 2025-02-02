import socket
import json
import pandas as pd
import time
import os
import warnings
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
warnings.simplefilter('ignore')


previous_data = {}  # store previous sensor data for movement comparison


# Function to determine direction and return a label for CSV output
def determine_direction(df):
    try:
        # Get the latest valid row
        latest_data = df.iloc[-1]
        
        # Store differences for all fingers
        x_diffs_all = []
        y_diffs_all = []
        
        for finger in ["Index", "Middle", "Ring"]:
            try:
                # Get positions
                proximal_x = float(latest_data[f"left{finger}Proximal_positionX"])
                proximal_y = float(latest_data[f"left{finger}Proximal_positionY"])
                distal_x = float(latest_data[f"left{finger}Distal_positionX"])
                distal_y = float(latest_data[f"left{finger}Distal_positionY"])
                
                # Skip if any values are NaN
                if any(pd.isna([proximal_x, proximal_y, distal_x, distal_y])):
                    continue
                
                # Calculate differences
                x_diff = distal_x - proximal_x
                y_diff = distal_y - proximal_y
                
                x_diffs_all.append(x_diff)
                y_diffs_all.append(y_diff)
                
                # Print for debugging
                print(f"{finger} X diff: {x_diff:.3f}, Y diff: {y_diff:.3f}")
                
            except (ValueError, TypeError) as e:
                print(f"Error processing {finger}: {str(e)}")
                continue
        
        # If we have valid differences, determine direction based on average movement
        if x_diffs_all and y_diffs_all:
            avg_x_diff = sum(x_diffs_all) / len(x_diffs_all)
            avg_y_diff = sum(y_diffs_all) / len(y_diffs_all)
            
            print(f"Average X diff: {avg_x_diff:.3f}, Y diff: {avg_y_diff:.3f}")
            
            # Define thresholds based on the data analysis
            x_threshold = 0.03
            y_threshold = 0.02
            
            # Check for the strongest direction
            if abs(avg_x_diff) > abs(avg_y_diff):
                # Horizontal movement is stronger
                if avg_x_diff > x_threshold:
                    return 4  # right
                elif avg_x_diff < -x_threshold:
                    return 3  # left
            else:
                # Vertical movement is stronger
                if avg_y_diff > y_threshold:
                    return 1  # up
                elif avg_y_diff < -y_threshold:
                    return 2  # down
                
    except Exception as e:
        print(f"Error processing DataFrame: {str(e)}")
        
    return None

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

   # This part needs to be aligned properly OUTSIDE the if-statement
#    direction_data = determine_direction(body)  # Call function once per frame
#    print("Direction:", direction_data)

   for body_part in body_parts:
        if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
            b = body.get(body_part, {})
            if "position" in b and "rotation" in b:
                position = b["position"]
                rotation = b["rotation"]
                body_data.extend([position.get("x", 0), position.get("y", 0), position.get("z", 0),
                                  rotation.get("x", 0), rotation.get("y", 0), rotation.get("z", 0), rotation.get("w", 0)])
   body_data.append(direction_data)
   body_data.append(time.strftime("%Y-%m-%d %H:%M:%S"))

   if body_data:  # Ensure that body_data is not empty
        return body_data, column_names, noHeader
   else:
        print("Body data is empty, nothing to save.")
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

def setup_robot_connection(host='172.16.164.131', port=1025):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        print(f"Connected to RobotStudio at {host}:{port}")
        return client
    except Exception as e:
        print(f"Failed to connect to RobotStudio: {e}")
        return None

def send_direction_to_robot(client, direction):
    if client and direction is not None:
        try:
            # Simply convert the direction to string
            message = f"{direction}\n"
            client.send(message.encode())
            print(f"Sent direction {direction} to RobotStudio")
        except Exception as e:
            print(f"Failed to send direction: {e}")
            return False
    return True

def main():
    # UDP setup for glove data
    UDP_IP = "127.0.0.1"
    UDP_PORT = 14043
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    
    # TCP setup for RobotStudio
    robot_client = setup_robot_connection(host='172.16.164.131', port=1025)
    if not robot_client:
        print("Failed to connect to RobotStudio. Exiting.")
        return
    
    print("Setting preset position...")
    df = set_preset_position()
    df.to_csv("preset_data.csv", index=False)
    print("Preset data recorded in 'preset_data.csv'.")
    
    column_names = []
    noHeader = True
    first_write = True
    
    input("Press Enter to start continuous data recording...")
    
    try:
        while True:
            print("Recording data for 5 seconds...")
            start_time = time.time()
            
            while time.time() - start_time < 5:
                sock.settimeout(0.1)
                try:
                    data, addr = sock.recvfrom(65000)
                    data = data.decode("utf-8")
                    
                    live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
                    if live_df_data is not None:
                        live_df = pd.DataFrame([live_df_data], columns=column_names)
                        live_df.to_csv("live_data.csv", mode='a', header=first_write, index=False)
                        first_write = False
                        
                        # Determine direction and send to RobotStudio
                        direction = determine_direction(live_df)
                        if direction is not None:
                            print("Direction:", direction)
                            if not send_direction_to_robot(robot_client, direction):
                                # Try to reconnect if sending failed
                                robot_client = setup_robot_connection(host='172.16.164.131', port=1025)
                            
                except socket.timeout:
                    continue
            
            print("Waiting for 5 seconds...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Data recording stopped.")
        print("Final data saved to live_data.csv.")
        if robot_client:
            robot_client.close()

if __name__ == "__main__":
    main()