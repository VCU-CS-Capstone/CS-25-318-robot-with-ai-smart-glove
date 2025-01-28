import socket
import time

"""
Note: If you have issues on the Robot Studio side of this program, check RobotStudio
firewall settings. Make sure RobotStudio program is running before running this one.
"""

def send_numbers():
    # Create TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Robot IP address and port (update these)
    robot_ip = '172.16.164.131'  # VM IP, changes based on who's running it
    port = 1025

    try:
        # Connect to the robot
        sock.connect((robot_ip, port))
        print(f"Connected to {robot_ip}:{port}")
        
        # Send numbers 1-6, each three times
        """
        Key:
            # 1 Move Up
            # 2 Move Down
            # 3 Move Left
            # 4 Move Right
            # 5 Move Back
            # 6 Move Forward
        """
        for number in range(1, 7):
            # Send the same number three times
            for _ in range(3):
                # Convert number to string and encode
                message = str(number).encode()
                sock.send(message)
                print(f"Sent: {number}")
                time.sleep(1)  # Wait 1 second between sends
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing connection")
        sock.close()

if __name__ == "__main__":
    send_numbers()
