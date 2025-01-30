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
def determine_direction(body_data):
    
    direction = []

    # List of fingers to evaluate
    fingers = ["Index", "Middle", "Ring"]

    # Evaluate positions for each finger
    for finger in fingers:
        #for part in ["Proximal", "Medial", "Distal"]:       #finger position            # has the name of the finger and the finger part
        proximal_part = f"left{finger}Proximal"         #
        distal_part = f"left{finger}Distal"

        try:
            proximal_position = body_data[proximal_part]["_position"]
            distal_position = body_data[distal_part]["_position"]
            #leftIndexProximal_positionZ

            # calculate difference
            # if proximal_position and distal_position:
            x_diffs = distal_position["X"] - proximal_position["X"]
            y_diffs = distal_position["Y"] - proximal_position["Y"]
                #z_diffs = distal_position["z"] - proximal_position["z"]

            #Create a function that has a condition statement that states 
            # when X values of the index, middle, ring, and pinky are positive (0.07 - 0.09) it prints "Right" and
            if 0.07 < x_diffs < 0.09:
                return 4     #4 for right
            # when X values of the index, middle, ring, and pinky are negative (-0.07 to -0.09) it prints "Left" and
            elif -0.09 < x_diffs < -0.07:
                return 3     #3 for left
            # when Y values of the index, middle, ring, and pinky are negative (-0.07 to -0.09) it prints "down" and 
            if -0.09 < y_diffs < -0.07:
                return 2     #2 for down
            # when Y values of the index, middle, ring, and pinky are positive (0.07 - 0.09) it prints "up" and
            elif 0.07 < y_diffs < 0.09:
                return 1     #1 for up
            # # when Z values of the index, middle, ring, and pinky are positive (-0.07 to -0.09) it prints "forward" and 
            # if x_diffs > 0.07 & x_diffs < 0.09:
            #     print ("1")     #1 for right
            # # when Z values of the index, middle, ring, and pinky are negative (-0.07 to -0.09) it prints "backward" 
            # if x_diffs > 0.07 & x_diffs < 0.09:
            #     print ("1")     #1 for right
            # # the print statements should be printed on the direction column of the data OR print out in the terminal
            # if x_diffs > 0.07 & x_diffs < 0.09:
            #     print ("1")     #1 for right
        except KeyError:
            print(f"Data for {finger} part missing.")
            continue
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
   
#    return body_data, column_names, noHeader

  
#    for body_part in body_parts:
#        if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
#            b = body.get(body_part, {})
#            if "position" in b and "rotation" in b:
#                if "Distal" in body_part or "Proximal" in body_part:
#                    finger_part = ''.join([ch for ch in body_part if ch.isalpha()])
#                    distal_part = finger_part + "Distal"
#                    proximal_part = finger_part + "Proximal"

#                    print("Direction", determine_direction(body))


                #    if distal_part in body and proximal_part in body:
                #        distal_pos = body[distal_part]["position"]
                #        proximal_pos = body[proximal_part]["position"]


                #        if distal_pos and proximal_pos:
                #            diff_x = distal_pos["x"] - proximal_pos["x"]
                #            diff_y = distal_pos["y"] - proximal_pos["y"]

                #            # Determine direction based on x and y values
                #            direction_data = determine_direction(body)


#    if body_data:
#        body_data.append(direction_data)  # Append the direction to the data row
#        body_data.append(time.strftime("%Y-%m-%d %H:%M:%S"))  # Append the timestamp to the data row
#        df = pd.DataFrame([body_data], columns=column_names)
#        return df, column_names, noHeader
#    else:
#        return None, column_names, noHeader

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
               #print("live_df_data:", live_df_data)  # Debug print
               live_df = pd.DataFrame([live_df_data], columns=column_names)
            #    print("Column names:", column_names)
            #    print("DataFrame created:", live_df)  # Check the DataFrame
               live_df.to_csv("live_data.csv", mode='a', header=first_write, index=False)
               first_write = False
            
            # Determine the direction
           direction = determine_direction(live_df)
           print("Direction:", direction)


           # 5-second waiting period
           print("Waiting for 5 seconds...")

           time.sleep(5)


   except KeyboardInterrupt:
       print("Data recording stopped.")
       print("Final data saved to live_data.csv.")


if __name__ == "__main__":
   main()