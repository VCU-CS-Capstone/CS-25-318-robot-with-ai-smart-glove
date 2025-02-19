

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
                proximal_x = float(latest_data[f"right{finger}Proximal_positionX"])
                proximal_y = float(latest_data[f"right{finger}Proximal_positionY"])
                distal_x = float(latest_data[f"right{finger}Distal_positionX"])
                distal_y = float(latest_data[f"right{finger}Distal_positionY"])
                
                # Skip if any values are NaN
                if any(pd.isna([proximal_x, proximal_y, distal_x, distal_y])):
                    continue
                
                # Calculate differences
                x_diff = distal_x - proximal_x
                y_diff = distal_y - proximal_y
                
                x_diffs_all.append(x_diff)
                y_diffs_all.append(y_diff)
                
                # Print for debugging
                #print(f"{finger} X diff: {x_diff:.3f}, Y diff: {y_diff:.3f}")
                
            except (ValueError, TypeError) as e:
                print(f"Error processing {finger}: {str(e)}")
                continue
        
        # If we have valid differences, determine direction based on average movement
        if x_diffs_all and y_diffs_all:
            avg_x_diff = sum(x_diffs_all) / len(x_diffs_all)
            avg_y_diff = sum(y_diffs_all) / len(y_diffs_all)
            
            #print(f"Average X diff: {avg_x_diff:.3f}, Y diff: {avg_y_diff:.3f}")
            
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
           if "right" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
               another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
                               body_part + "_rotation_x", body_part + "_rotation_y", body_part + "_rotation_z", body_part + "_rotation_w"]
               column_names.extend(another_temp)
       column_names.extend(["direction", "timestamp"])  # Add direction and timestamp columns
       noHeader = False

   # This part needs to be aligned properly OUTSIDE the if-statement
#    direction_data = determine_direction(body)  # Call function once per frame
#    print("Direction:", direction_data)

   for body_part in body_parts:
        if "right" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
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

def setup_tcp_connection(target_ip, target_port):
    """Set up TCP connection similar to sender.py"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"Attempting to connect to {target_ip}:{target_port}")
        sock.connect((target_ip, target_port))
        print(f"Connected to {target_ip}:{target_port}")
        return sock
    except ConnectionRefusedError:
        print("Connection refused - make sure target device is listening!")
        return None
    except Exception as e:
        print(f"Error connecting: {e}")
        return None

def send_direction(tcp_sock, direction):
    """Send direction number over TCP connection"""
    if tcp_sock and direction is not None:
        try:
            message = str(direction).encode()
            tcp_sock.send(message)
            print(f"Sent direction {direction} over TCP")
            return True
        except Exception as e:
            print(f"Error sending direction: {e}")
            return False
    return False

def main():
    # UDP setup for Rokoko glove
    UDP_IP = "127.0.0.1"
    UDP_PORT = 14043
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_sock.bind((UDP_IP, UDP_PORT))
    
    # TCP setup for sending directions
    target_ip = '172.20.10.12'  # Update with your target IP
    target_port = 5001  # Update with your target port
    tcp_sock = setup_tcp_connection(target_ip, target_port)
    
    print("Setting preset position...")
    df = set_preset_position()
    df.to_csv("preset_data.csv", index=False)
    print("Preset data recorded in 'preset_data.csv'.")
    
    column_names = []
    noHeader = True
    first_write = True
    
    # Try to reconnect TCP if initial connection failed
    if tcp_sock is None:
        print("Initial TCP connection failed. Will try to reconnect later.")
    
    input("Press Enter to start continuous data recording...")
    
    last_direction = None
    last_direction_time = 0
    cooldown_period = 1.0  # Seconds to wait before sending the same direction again
    
    try:
        while True:
            current_time = time.time()
            
            # Try to reconnect TCP if not connected
            if tcp_sock is None and current_time - last_direction_time > 5:
                print("Attempting to reconnect TCP...")
                tcp_sock = setup_tcp_connection(target_ip, target_port)
                last_direction_time = current_time
            
            # Set a timeout for sock.recvfrom to prevent blocking
            udp_sock.settimeout(0.1)
            try:
                data, addr = udp_sock.recvfrom(65000)
                data = data.decode("utf-8")
                
                live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
                if live_df_data is not None:
                    live_df = pd.DataFrame([live_df_data], columns=column_names)
                    live_df.to_csv("live_data.csv", mode='a', header=first_write, index=False)
                    first_write = False
                    
                    # Determine the direction
                    direction = determine_direction(live_df)
                    if direction is not None:  # Only process if we detect a direction
                        print("Direction detected:", direction)
                        
                        # Only send if it's a different direction or enough time has passed
                        if direction != last_direction or (current_time - last_direction_time > cooldown_period):
                            if tcp_sock:
                                if send_direction(tcp_sock, direction):
                                    last_direction = direction
                                    last_direction_time = current_time
                                else:
                                    # Connection might be broken, set to None to trigger reconnection
                                    tcp_sock = None
                            
            except socket.timeout:
                continue  # If no data received, continue the loop
            except Exception as e:
                print(f"Error in main loop: {e}")
                # If TCP socket error, try to reconnect next iteration
                if "tcp" in str(e).lower() or "connection" in str(e).lower():
                    tcp_sock = None
            
    except KeyboardInterrupt:
        print("Data recording stopped.")
        print("Final data saved to live_data.csv.")
        if tcp_sock:
            tcp_sock.close()
        udp_sock.close()

if __name__ == "__main__":
    main()

# previous_data = {}  # store previous sensor data for movement comparison


# # Function to determine direction and return a label for CSV output
# def determine_direction(df):
#     try:
#         # Get the latest valid row
#         latest_data = df.iloc[-1]
        
#         # Store differences for all fingers
#         x_diffs_all = []
#         y_diffs_all = []
        
#         for finger in ["Index", "Middle", "Ring"]:
#             try:
#                 # Get positions
#                 proximal_x = float(latest_data[f"right{finger}Proximal_positionX"])
#                 proximal_y = float(latest_data[f"right{finger}Proximal_positionY"])
#                 distal_x = float(latest_data[f"right{finger}Distal_positionX"])
#                 distal_y = float(latest_data[f"right{finger}Distal_positionY"])
                
#                 # Skip if any values are NaN
#                 if any(pd.isna([proximal_x, proximal_y, distal_x, distal_y])):
#                     continue
                
#                 # Calculate differences
#                 x_diff = distal_x - proximal_x
#                 y_diff = distal_y - proximal_y
                
#                 x_diffs_all.append(x_diff)
#                 y_diffs_all.append(y_diff)
                
#             except (ValueError, TypeError) as e:
#                 print(f"Error processing {finger}: {str(e)}")
#                 continue
        
#         # If we have valid differences, determine direction based on average movement
#         if x_diffs_all and y_diffs_all:
#             avg_x_diff = sum(x_diffs_all) / len(x_diffs_all)
#             avg_y_diff = sum(y_diffs_all) / len(y_diffs_all)
            
#             # Define thresholds based on the data analysis
#             x_threshold = 0.03
#             y_threshold = 0.02
            
#             # Check for the strongest direction
#             if abs(avg_x_diff) > abs(avg_y_diff):
#                 # Horizontal movement is stronger
#                 if avg_x_diff > x_threshold:
#                     return 4  # right
#                 elif avg_x_diff < -x_threshold:
#                     return 3  # left
#             else:
#                 # Vertical movement is stronger
#                 if avg_y_diff > y_threshold:
#                     return 1  # up
#                 elif avg_y_diff < -y_threshold:
#                     return 2  # down
                
#     except Exception as e:
#         print(f"Error processing DataFrame: {str(e)}")
        
#     return None

# def write_direction_to_file(direction, filename="/Users/erinanderson/Dropbox/RobotSharing/robot_commands2.txt"):
#     """Write direction to file and flush immediately"""
#     if direction is not None:
#         try:
#             with open(filename, 'a') as f:
#                 f.write(f"{direction}\n")
#                 f.flush()  # Force write to disk
#                 os.fsync(f.fileno())  # Ensure it's written to disk
#         except Exception as e:
#             print(f"Error writing to file: {e}")

# def live_data_to_df(data, column_names, noHeader):
#     body_data = []  # List to store parsed data for each body part
#     direction_data = None  # Variable to store direction for CSV output

#     # Attempt to decode received JSON data
#     try:
#         d = json.loads(data)
#     except json.JSONDecodeError:
#         print("Error decoding JSON data")
#         return None, column_names, noHeader
    
#     if "scene" not in d or "actors" not in d["scene"] or len(d["scene"]["actors"]) == 0:
#         print("No actors or scene data found")
#         return None, column_names, noHeader
    
#     body = d["scene"]["actors"][0].get("body", {})
#     if not body:
#         print("No body data available")
#         return None, column_names, noHeader

#     body_parts = list(body.keys())
#     # Add header if it's the first run
#     if noHeader:
#         for body_part in body_parts:
#             if "right" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
#                 another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
#                               body_part + "_rotation_x", body_part + "_rotation_y", body_part + "_rotation_z", body_part + "_rotation_w"]
#                 column_names.extend(another_temp)
#         column_names.extend(["direction", "timestamp"])  # Add direction and timestamp columns
#         noHeader = False

#     for body_part in body_parts:
#         if "right" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
#             b = body.get(body_part, {})
#             if "position" in b and "rotation" in b:
#                 position = b["position"]
#                 rotation = b["rotation"]
#                 body_data.extend([position.get("x", 0), position.get("y", 0), position.get("z", 0),
#                                 rotation.get("x", 0), rotation.get("y", 0), rotation.get("z", 0), rotation.get("w", 0)])
#     body_data.append(direction_data)
#     body_data.append(time.strftime("%Y-%m-%d %H:%M:%S"))

#     if body_data:  # Ensure that body_data is not empty
#         return body_data, column_names, noHeader
#     else:
#         print("Body data is empty, nothing to save.")
#         return None, column_names, noHeader

# def set_preset_position():
#     # Define a preset position
#     preset_position = [0.0, 0.0, 0.0,   # positionX, positionY, positionZ
#                       0.0, 0.0, 0.0, 1.0]  # rotation_x, rotation_y, rotation_z, rotation_w

#     # Create a DataFrame with the correct columns
#     columns = ["positionX", "positionY", "positionZ", "rotation_x", "rotation_y", "rotation_z", "rotation_w"]
#     df = pd.DataFrame(columns=columns)

#     # Add the preset position to the DataFrame
#     df.loc[len(df)] = preset_position
#     return df

# def main():
#     # Clear the robotcommands2.txt file at startup
#     with open("/Users/erinanderson/Dropbox/RobotSharing/robot_commands2.txt", 'w') as f:
#         f.write("")  # Clear the file
#         f.flush()    # Force write to disk
#         os.fsync(f.fileno())  # Ensure it's written to disk
    
#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 14043
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     sock.bind((UDP_IP, UDP_PORT))
    
#     print("Setting preset position...")
#     df = set_preset_position()
#     df.to_csv("preset_data.csv", index=False)
#     print("Preset data recorded in 'preset_data.csv'.")
    
#     column_names = []
#     noHeader = True
#     first_write = True
    
#     input("Press Enter to start continuous data recording...")
    
#     try:
#         # Set non-blocking socket
#         sock.setblocking(False)
        
#         while True:
#             try:
#                 data, addr = sock.recvfrom(65000)
#                 data = data.decode("utf-8")
                
#                 live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
#                 if live_df_data is not None:
#                     live_df = pd.DataFrame([live_df_data], columns=column_names)
#                     live_df.to_csv("live_data.csv", mode='a', header=first_write, index=False)
#                     first_write = False
                    
#                     # Determine the direction and write to file immediately
#                     direction = determine_direction(live_df)
#                     if direction is not None:  # Only write if we detect a direction
#                         print("Direction:", direction)
#                         write_direction_to_file(direction)
                        
#             except BlockingIOError:
#                 # No data available right now, continue immediately
#                 continue
#             except socket.error as e:
#                 # Handle other socket errors if needed
#                 print(f"Socket error: {e}")
#                 continue
                
#             # Small sleep to prevent CPU overuse
#             time.sleep(0.01)  # 10ms delay
            
#     except KeyboardInterrupt:
#         print("Data recording stopped.")
#         print("Final data saved to live_data.csv.")

# if __name__ == "__main__":
#     main()

# # # Function to determine direction and return a label for CSV output
# # def determine_direction(df):
# #     try:
# #         # Get the latest valid row
# #         latest_data = df.iloc[-1]
        
# #         # Store differences for all fingers
# #         x_diffs_all = []
# #         y_diffs_all = []
        
# #         for finger in ["Index", "Middle", "Ring"]:
# #             try:
# #                 # Get positions
# #                 proximal_x = float(latest_data[f"left{finger}Proximal_positionX"])
# #                 proximal_y = float(latest_data[f"left{finger}Proximal_positionY"])
# #                 distal_x = float(latest_data[f"left{finger}Distal_positionX"])
# #                 distal_y = float(latest_data[f"left{finger}Distal_positionY"])
                
# #                 # Skip if any values are NaN
# #                 if any(pd.isna([proximal_x, proximal_y, distal_x, distal_y])):
# #                     continue
                
# #                 # Calculate differences
# #                 x_diff = distal_x - proximal_x
# #                 y_diff = distal_y - proximal_y
                
# #                 x_diffs_all.append(x_diff)
# #                 y_diffs_all.append(y_diff)
                
# #             except (ValueError, TypeError) as e:
# #                 print(f"Error processing {finger}: {str(e)}")
# #                 continue
        
# #         # If we have valid differences, determine direction based on average movement
# #         if x_diffs_all and y_diffs_all:
# #             avg_x_diff = sum(x_diffs_all) / len(x_diffs_all)
# #             avg_y_diff = sum(y_diffs_all) / len(y_diffs_all)
            
# #             # Define thresholds based on the data analysis
# #             x_threshold = 0.03
# #             y_threshold = 0.02
            
# #             # Check for the strongest direction
# #             if abs(avg_x_diff) > abs(avg_y_diff):
# #                 # Horizontal movement is stronger
# #                 if avg_x_diff > x_threshold:
# #                     return 4  # right
# #                 elif avg_x_diff < -x_threshold:
# #                     return 3  # left
# #             else:
# #                 # Vertical movement is stronger
# #                 if avg_y_diff > y_threshold:
# #                     return 1  # up
# #                 elif avg_y_diff < -y_threshold:
# #                     return 2  # down
                
# #     except Exception as e:
# #         print(f"Error processing DataFrame: {str(e)}")
        
# #     return None

# # def write_direction_to_file(direction, filename="/Users/erinanderson/Dropbox/RobotSharing/robot_commands2.txt"):
# #     """Write direction to file, ensuring it's a new line"""
# #     if direction is not None:
# #         try:
# #             with open(filename, 'a') as f:
# #                 f.write(f"{direction}\n")
# #         except Exception as e:
# #             print(f"Error writing to file: {e}")

# # def live_data_to_df(data, column_names, noHeader):
# #     body_data = []  # List to store parsed data for each body part
# #     direction_data = None  # Variable to store direction for CSV output

# #     # Attempt to decode received JSON data
# #     try:
# #         d = json.loads(data)
# #     except json.JSONDecodeError:
# #         print("Error decoding JSON data")
# #         return None, column_names, noHeader
    
# #     if "scene" not in d or "actors" not in d["scene"] or len(d["scene"]["actors"]) == 0:
# #         print("No actors or scene data found")
# #         return None, column_names, noHeader
    
# #     body = d["scene"]["actors"][0].get("body", {})
# #     if not body:
# #         print("No body data available")
# #         return None, column_names, noHeader

# #     body_parts = list(body.keys())
# #     # Add header if it's the first run
# #     if noHeader:
# #         for body_part in body_parts:
# #             if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
# #                 another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
# #                               body_part + "_rotation_x", body_part + "_rotation_y", body_part + "_rotation_z", body_part + "_rotation_w"]
# #                 column_names.extend(another_temp)
# #         column_names.extend(["direction", "timestamp"])  # Add direction and timestamp columns
# #         noHeader = False

# #     for body_part in body_parts:
# #         if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
# #             b = body.get(body_part, {})
# #             if "position" in b and "rotation" in b:
# #                 position = b["position"]
# #                 rotation = b["rotation"]
# #                 body_data.extend([position.get("x", 0), position.get("y", 0), position.get("z", 0),
# #                                 rotation.get("x", 0), rotation.get("y", 0), rotation.get("z", 0), rotation.get("w", 0)])
# #     body_data.append(direction_data)
# #     body_data.append(time.strftime("%Y-%m-%d %H:%M:%S"))

# #     if body_data:  # Ensure that body_data is not empty
# #         return body_data, column_names, noHeader
# #     else:
#         print("Body data is empty, nothing to save.")
#         return None, column_names, noHeader

# def set_preset_position():
#     # Define a preset position
#     preset_position = [0.0, 0.0, 0.0,   # positionX, positionY, positionZ
#                       0.0, 0.0, 0.0, 1.0]  # rotation_x, rotation_y, rotation_z, rotation_w

#     # Create a DataFrame with the correct columns
#     columns = ["positionX", "positionY", "positionZ", "rotation_x", "rotation_y", "rotation_z", "rotation_w"]
#     df = pd.DataFrame(columns=columns)

#     # Add the preset position to the DataFrame
#     df.loc[len(df)] = preset_position
#     return df

# def main():
#     # Clear the robotcommands2.txt file at startup
#     with open("robotcommands2.txt", 'w') as f:
#         f.write("")  # Clear the file
    
#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 14043
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     sock.bind((UDP_IP, UDP_PORT))
    
#     print("Setting preset position...")
#     df = set_preset_position()
#     df.to_csv("preset_data.csv", index=False)
#     print("Preset data recorded in 'preset_data.csv'.")
    
#     column_names = []
#     noHeader = True
#     first_write = True
    
#     input("Press Enter to start continuous data recording...")
    
#     try:
#         while True:
#             start_time = time.time()
            
#             # Collect data for 5 seconds
#             while time.time() - start_time < 5:
#                 # Set a timeout for sock.recvfrom to prevent blocking
#                 sock.settimeout(0.1)
#                 try:
#                     data, addr = sock.recvfrom(65000)
#                     data = data.decode("utf-8")
                    
#                     live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
#                     if live_df_data is not None:
#                         live_df = pd.DataFrame([live_df_data], columns=column_names)
#                         live_df.to_csv("live_data.csv", mode='a', header=first_write, index=False)
#                         first_write = False
                        
#                         # Determine the direction and write to file
#                         direction = determine_direction(live_df)
#                         if direction is not None:  # Only write if we detect a direction
#                             print("Direction:", direction)
#                             write_direction_to_file(direction)
                            
#                 except socket.timeout:
#                     continue  # If no data received, continue the loop
            
#     except KeyboardInterrupt:
#         print("Data recording stopped.")
#         print("Final data saved to live_data.csv.")

# if __name__ == "__main__":
#     main()

# def setup_robot_connection(host='172.16.164.131', port=1025):
#     try:
#         client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         client.connect((host, port))
#         print(f"Connected to RobotStudio at {host}:{port}")
#         return client
#     except Exception as e:
#         print(f"Failed to connect to RobotStudio: {e}")
#         return None

# def send_direction_to_robot(client, direction):
#     if client and direction is not None:
#         try:
#             # Simply convert the direction to string
#             message = f"{direction}\n"
#             client.send(message.encode())
#             print(f"Sent direction {direction} to RobotStudio")
#         except Exception as e:
#             print(f"Failed to send direction: {e}")
#             return False
#     return True

# def main():
#     # UDP setup for glove data
#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 14043
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     sock.bind((UDP_IP, UDP_PORT))
    
#     # TCP setup for RobotStudio
#     robot_client = setup_robot_connection(host='172.16.164.131', port=1025)
#     if not robot_client:
#         print("Failed to connect to RobotStudio. Exiting.")
#         return
    
#     print("Setting preset position...")
#     df = set_preset_position()
#     df.to_csv("preset_data.csv", index=False)
#     print("Preset data recorded in 'preset_data.csv'.")
    
#     column_names = []
#     noHeader = True
#     first_write = True
    
#     input("Press Enter to start continuous data recording...")
    
#     try:
#         while True:
#             print("Recording data for 5 seconds...")
#             start_time = time.time()
            
#             while time.time() - start_time < 5:
#                 sock.settimeout(0.1)
#                 try:
#                     data, addr = sock.recvfrom(65000)
#                     data = data.decode("utf-8")
                    
#                     live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
#                     if live_df_data is not None:
#                         live_df = pd.DataFrame([live_df_data], columns=column_names)
#                         live_df.to_csv("live_data.csv", mode='a', header=first_write, index=False)
#                         first_write = False
                        
#                         # Determine direction and send to RobotStudio
#                         direction = determine_direction(live_df)
#                         if direction is not None:
#                             print("Direction:", direction)
#                             if not send_direction_to_robot(robot_client, direction):
#                                 # Try to reconnect if sending failed
#                                 robot_client = setup_robot_connection(host='172.16.164.131', port=1025)
                            
#                 except socket.timeout:
#                     continue
            
#             print("Waiting for 5 seconds...")
#             time.sleep(5)
            
#     except KeyboardInterrupt:
#         print("Data recording stopped.")
#         print("Final data saved to live_data.csv.")
#         if robot_client:
#             robot_client.close()

# if __name__ == "__main__":
#     main()