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
def determine_direction(body):
    
    direction = []

    # List of fingers to evaluate
    fingers = ["Index", "Middle", "Ring", "Little"]

    # Initialize direction counters
    x_positive_count = 0
    x_negative_count = 0
    y_positive_count = 0
    y_negative_count = 0
    z_positive_count = 0
    z_negative_count = 0

    # Evaluate positions for each finger
    for finger in fingers:
        for part in ["Proximal", "Medial", "Distal"]:
            finger_part = f"left{finger}{part}"
            if finger_part in body:
                position = body[finger_part].get("position", {})
                x = position.get("x", 0)
                y = position.get("y", 0)
                z = position.get("z", 0)

                # Check X values
                if 0.07 <= x <= 0.09:
                    x_positive_count += 1
                elif -0.09 <= x <= -0.07:
                    x_negative_count += 1

                # Check Y values
                if 0.07 <= y <= 0.09:
                    y_positive_count += 1
                elif -0.09 <= y <= -0.07:
                    y_negative_count += 1

                # Check Z values
                if 0.07 <= z <= 0.09:
                    z_positive_count += 1
                elif -0.09 <= z <= -0.07:
                    z_negative_count += 1

    # Determine the dominant direction
    if x_positive_count >= 4:
        direction.append("Right")
    elif x_negative_count >= 4:
        direction.append("Left")

    if y_positive_count >= 4:
        direction.append("Up")
    elif y_negative_count >= 4:
        direction.append("Down")

    if z_positive_count >= 4:
        direction.append("Forward")
    elif z_negative_count >= 4:
        direction.append("Backward")

    # Return a joined direction string or "None" if no direction
    return ", ".join(direction) if direction else "None"
#    if diff_y > 0:
#        return "up"
#    elif diff_y < 0:
#        return "down"
#    elif diff_x > 0:
#        return "right"
#    elif diff_x < 0:
#        return "left"
#    return None
    # body = []
    # direction = []

    # # Finger parts to evaluate
    # fingers = ["Index", "Middle", "Ring", "Little"]

    # for finger in fingers:
    #     if finger in body:
    #         pos = body[finger].get("position", {})
    #         if pos:
    #             x, y, z = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)

    #             if 0.07 <= x <= 0.09:
    #                 print("Right")
    #                 direction.append("Right")
    #             elif -0.09 <= x <= -0.07:
    #                 print("Left")
    #                 direction.append("Left")

    #             if 0.07 <= y <= 0.09:
    #                 print("Up")
    #                 direction.append("Up")
    #             elif -0.09 <= y <= -0.07:
    #                 print("Down")
    #                 direction.append("Down")

    #             if 0.07 <= z <= 0.09:
    #                 print("Forward")
    #                 direction.append("Forward")
    #             elif -0.09 <= z <= -0.07:
    #                 print("Left")
    #                 direction.append("Backward")

    # # Join directions for clarity, or return "None" if no valid direction
    # #return ", ".join(direction) if direction else "None"

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
                           direction_data = determine_direction(body)


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

# Create a function that has a condition statement that states 
# when X values of the index, middle, ring, and pinky are positive (0.07 - 0.09) it prints "Right" and 
# when X values of the index, middle, ring, and pinky are negative (-0.07 to -0.09) it prints "Left" and
# when Y values of the index, middle, ring, and pinky are positive (0.07 - 0.09) it prints "up" and
# when Y values of the index, middle, ring, and pinky are negative (-0.07 to -0.09) it prints "down" and 
# when Z values of the index, middle, ring, and pinky are positive (-0.07 to -0.09) it prints "forward" and 
# when Z values of the index, middle, ring, and pinky are negative (-0.07 to -0.09) it prints "backward" 
# the print statements should be printed on the direction column of the data OR print out in the terminal

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
            #    live_df_data.to_csv("live_data.csv", mode='a', header=first_write, index=False)
            #    print("Direction:", live_df_data.iloc[0]["direction"])
            #    first_write = False
               print("Direction:", live_df_data["direction"].values[0])  # Use .values[0] to extract the first value
               live_df_data.to_csv("live_data.csv", mode='a', header=first_write, index=False)
               first_write = False



           # 5-second waiting period
           print("Waiting for 5 seconds...")

           #time.sleep(5)


   except KeyboardInterrupt:
       print("Data recording stopped.")
       print("Final data saved to live_data.csv.")


if __name__ == "__main__":
   main()


# warnings.simplefilter('ignore')

# previous_data = {}  # store previous sensor data for movement comparison
# clf = joblib.load("finger_direction_model.joblib")

# # Function to determine direction and return a label for CSV output
# def determine_direction(diff_x, diff_y):
#     if diff_y > 0:
#         return "up"
#     elif diff_y < 0:
#         return "down"
#     elif diff_x > 0:
#         return "right"
#     elif diff_x < 0:
#         return "left"
#     return None

# def predict_direction(rotation_data):
#     return clf.predict([rotation_data])[0]  # Predicts and returns a direction

# #converting quaternion to euler angles
# def quaternion_to_euler(x, y, z, w):

#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll_x = round(math.atan2(t0, t1), 3)

#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch_y = round(math.asin(t2), 3)

#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw_z = round(math.atan2(t3, t4), 3)

#     return roll_x, pitch_y, yaw_z  # in radians. to make it degrees it would be = round(180/math.pi, 3)

# # Function to parse incoming live data, process it, and store in a DataFrame
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
#             if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
#                 another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
#                                 body_part + "_rotation_x", body_part + "_rotation_y", body_part + "_rotation_z", body_part + "_rotation_w"]
#                 column_names.extend(another_temp)
#         column_names.extend(["direction", "timestamp"])  # Add direction and timestamp columns
#         noHeader = False

#     for body_part in body_parts:
#         if "left" in body_part and any(part in body_part for part in ["Lower", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"]):
#             b = body.get(body_part, {})
#             if "position" in b and "rotation" in b:
#                 # if "Distal" in body_part and "Proximal" in body_part:
#                 #     finger_part = ''.join([ch for ch in body_part if ch.isalpha()])
#                 #     distal_part = finger_part + "Distal"
#                 #     proximal_part = finger_part + "Proximal"

#                 #     if distal_part in body and proximal_part in body:
#                 #         distal_pos = body[distal_part]["position"]
#                 #         proximal_pos = body[proximal_part]["position"]

#                 #         if distal_pos and proximal_pos:
#                 #             diff_x = distal_pos["x"] - proximal_pos["x"]
#                 #             diff_y = distal_pos["y"] - proximal_pos["y"]

#                 #             # Determine direction based on x and y values
#                 #             direction_data = determine_direction(diff_x, diff_y)
#                 rotation_data = [b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"], b["rotation"]["w"]]
#                 direction_data = predict_direction(rotation_data)  # Predict direction using the ML model
#                 temp = [b["position"]["x"], b["position"]["y"], b["position"]["z"],
#                         b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"], b["rotation"]["w"]]
#                 body_data.extend(temp)

#     if body_data:
#         body_data.append(direction_data)  # Append the direction to the data row
#         body_data.append(time.strftime("%Y-%m-%d %H:%M:%S"))  # Append the timestamp to the data row
#         df = pd.DataFrame([body_data], columns=column_names)
#         return df, column_names, noHeader
#     else:
#         return None, column_names, noHeader

# def set_preset_position():
#     # Define a preset position
#     preset_position = [0.0, 0.0, 0.0,   # positionX, positionY, positionZ
#                        0.0, 0.0, 0.0, 1.0]  # rotation_x, rotation_y, rotation_z, rotation_w

#     # Create a DataFrame with the correct columns
#     columns = ["positionX", "positionY", "positionZ", "rotation_x", "rotation_y", "rotation_z", "rotation_w"]
#     df = pd.DataFrame(columns=columns)

#     # Add the preset position to the DataFrame
#     df.loc[len(df)] = preset_position
#     return df

# # Main loop to receive data and process it
# def main():
#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 14043

#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     sock.bind((UDP_IP, UDP_PORT))

#     print("Setting preset position...")
#     df = set_preset_position()  # Set the initial position
#     df.to_csv("preset_data.csv", index=False)
#     print("Preset data recorded in 'preset_data.csv'.")

#     column_names = []
#     noHeader = True
#     first_write = True

#     input("Press Enter to start continuous data recording...")

#     try:
#         while True:
#             # Start 5-second recording period
#             print("Recording data for 5 seconds...")
#             data, addr = sock.recvfrom(65000)
#             data = data.decode("utf-8")
            
#             live_df_data, column_names, noHeader = live_data_to_df(data, column_names, noHeader)
            
#             if live_df_data is not None:
#                 live_df_data.to_csv("live_data.csv", mode='a', header=first_write, index=False)
#                 first_write = False

#             # 5-second waiting period
#             print("Waiting for 5 seconds...")
#             time.sleep(5)

#     except KeyboardInterrupt:
#         print("Data recording stopped.")
#         print("Final data saved to live_data.csv.")

# if __name__ == "__main__":
#     main()

