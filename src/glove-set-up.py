import socket
import json
import pandas as pd
import time
import math
import warnings
from data_to_df import live_data_to_df
warnings.simplefilter('ignore')


def live_data_to_df(t):


   noHeader = True
   column_names = []
   df = pd.DataFrame
   variance_df = pd.DataFrame
   print("moving...")


   for i in range(t):
       #TODO: for each loop, add the "body_data" list to a dataframe object. since the body data is going to
       # reset to have nothing in it each time, i have to add the row after each iteration of this for loop.
       # Then, i need to set the column names after each finger part and then include its roll, pitch and yaw instead of quaternion rotational units.


       body_data = []


       sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       sock.bind(("", 14043))
       d, addr = sock.recvfrom(65000)
       d = d.decode("utf-8")
       d = json.loads(d)


       # timestamp = d["scene"]


       body = d["scene"]["actors"][0]["body"]
       body_parts = list(body.keys())


       if noHeader:
           for body_part in body_parts:
               if "right" in body_part:
                   if "Lower" in body_part or "Hand" in body_part or "Thumb" in body_part or "Index" in body_part or "Middle" in body_part or "Ring" in body_part or "Little" in body_part:
                       another_temp = [body_part + "_positionX", body_part + "_positionY", body_part + "_positionZ",
                                       body_part + "rotation_x", body_part + "rotation_y", body_part + "rotation_z", body_part + "rotation_w"]
                       column_names.extend(another_temp)
           # print(column_names)
           df = pd.DataFrame(columns=column_names)
           variance_df = pd.DataFrame(columns=column_names)
           noHeader = False


       # print(body["leftIndexMedial"]["rotation"]["x"])
       for body_part in body_parts:
           if "right" in body_part:
               if "Lower" in body_part or "Hand" in body_part or "Thumb" in body_part or "Index" in body_part or "Middle" in body_part or "Ring" in body_part or "Little" in body_part:
                   b = body[body_part]
                   # r, p, y = quaternion_to_euler(b["rotation"]["x"], b["rotation"]["y"], b["rotation"]["z"],
                   #                               b["rotation"]["w"])
                   temp = [b["position"]["x"], b["position"]["y"], b["position"]["z"], b["rotation"]["x"],b["rotation"]["y"],
                           b["rotation"]["z"], b["rotation"]["w"]]
                   body_data.extend(temp)


       df.loc[len(df)] = body_data


   return df




def getSocketData(ip, port):
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.bind((ip, port))
   data, addr = sock.recvfrom(65000)
   data = data.decode("utf-8")
   data = json.loads(data)
   return data


def quaternion_to_euler(x, y, z, w):


   t0 = +2.0 * (w * x + y * z)
   t1 = +1.0 - 2.0 * (x * x + y * y)
   roll_x = round(math.atan2(t0, t1), 3)


   t2 = +2.0 * (w * y - z * x)
   t2 = +1.0 if t2 > +1.0 else t2
   t2 = -1.0 if t2 < -1.0 else t2
   pitch_y = round(math.asin(t2), 3)


   t3 = +2.0 * (w * z + x * y)
   t4 = +1.0 - 2.0 * (y * y + z * z)
   yaw_z = round(math.atan2(t3, t4), 3)


   return roll_x, pitch_y, yaw_z  # in radians. to make it degrees it would be = round(180/math.pi, 3)


def get_header(thedata):
   titles = []
   for values in thedata:
       if "right" in values:
           if "Thumb" in values or "Index" in values or "Middle" in values or "Ring" in values or "Little" in values:
               titles.append(values)
   return titles


# def get_stdev(df):
#     dev = df.std()
#     stdev_list = dev.to_list()
#
#     return stdev_list




continue_gesture = True
header = False
dfs = []


while continue_gesture:
   choice = input("Do you want to add a new gesture? (yes or no).\n")
   if choice == 'yes':
       continue_trials = True
       gesture_name = input("Okay. What is the name of the gesture?\n")
       trials = int(input("Okay. How many trials will you perform?\n"))
       duration = int(input("Okay. How long do you want perform the gesture for?(seconds)\nFor instance, 120, 1200?:\n"))


       count = 0
       while continue_trials:
           wait = input("Enter anything when ur ready to start.")
           time.sleep(1)
           print("3")
           time.sleep(1)
           print("2")
           time.sleep(1)
           print("1")
           time.sleep(1)
           print("now!")
           live_df_data = live_data_to_df(duration)


           # for column in live_df_data:
           #     if "position" in column:
           #         live_df_data[column] = live_df_data[column].apply(lambda x: x*100)
           live_df_data["Gesture"] = gesture_name
           # if not header:
           #     names = ["Gesture"]
           #     title_headers = get_header(live_df_data)
           #     names.extend(title_headers)
           #     writer.writerow(names)
           #     header = True


           # TODO: determine the standard dev of each value and add it to another csv file.


           # stdevs_list.insert(0, gesture_name)
           # writer.writerow(stdevs_list)
           count += 1
           trials -= 1


           print(live_df_data)
           dfs.append(live_df_data)


           if trials != 0:
               continue_trials = True
               print(f"Trial {count} is complete.")
               print("Okay. Get ready for the next trial.\n")
           else:
               continue_trials = False


   elif choice == 'no':
       continue_gesture = False
       combined_database = pd.concat(dfs, ignore_index=True)
       combined_database.to_csv("Final_gestures_hopefully.csv", index=False) #name the file here.
