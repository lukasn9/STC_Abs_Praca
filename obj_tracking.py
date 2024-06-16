from azure.iot.device import IoTHubDeviceClient
from azure.iot.device import Message
import cv2
from cv2 import line, putText, rectangle, FONT_HERSHEY_SIMPLEX
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import create_video_writer
import json
from numpy import linspace
import sys
import threading
from time import sleep
from ultralytics import YOLO

CONNECTION_STRING = "[Iot Hub connection string]"
CONFIDENCE_THRESHOLD = 0.75
MAX_AGE = 50
DARK_BLUE = (27, 45, 166)
WHITE = (255, 255, 255)

FIRST_GOAL_COORDS = [890, 255, 1070, 315, 5]
GOAL_1_HEIGHT = 135

first_name = True
first_frame = True

pass_cooldown = 0
shot_cooldown = 0

x_values = []
y_values  = []

pitch_x_coord = 0
pitch_y_coord = 0

ball_coords_1 = []
ball_coords_2 = []
ball_coords_x = []
ball_coords_y = []
message_content = ""
is_running = True

video_cap = ""
tracker = 0
program_start = ""
writer = ""
model = ""
device_client = ""

pass_num = 0
shot_num = 1
goal_num = 1

azure_messages_queue = []
azure_messages = []

coords_player_dict = {
    "player_id": 0,
    "x": 0,
    "y": 0,
    "passes": 0,
    "shots": 0,
    "goals": 0
}

coords_ball_dict = {
    "player_id": "Ball",
    "x": 0,
    "y": 0,
    "passes": 0,
    "shots": 0,
    "goals": 0
}

def detect_goal():
    global GOAL_1_HEIGHT
    global x_values
    global y_values
    global goal_num
    global ball_coords_x
    global ball_coords_y

    while is_running == True:
        if len(ball_coords_x) == 0 or len(ball_coords_y) == 0:
            sleep(2)
        else:
            j = 0
            for coord_x in ball_coords_x:
                for i in range(200):
                    if coord_x == 890 + i:
                        cur_y = y_values[i]
                        y = cur_y
                        while y <= (cur_y + GOAL_1_HEIGHT):
                            if ball_coords_y[j] == y:
                                goal_num += 1
                                print("Goal")
                            else:
                                y += 1
                ball_coords_x.pop(0)
                ball_coords_y.pop(0)
                j += 1

            ball_coords_x = []
            ball_coords_y = []

def run_sample(device_client):
    global azure_messages_queue
    global azure_messages
    global is_running

    while is_running == True:
        if len(azure_messages) == 0:
            sleep(5)
            azure_messages = azure_messages_queue
        else:
            for message_content in azure_messages:
                message = Message(message_content)

                try:
                    device_client.send_message(message)
                    print(f"Message sent: {message_content}")

                except Exception as ex:
                    print("Error occurred while sending message:", ex)
                    sys.exit()
                
                azure_messages_queue.pop(0)

            azure_messages = []

    azure_messages = azure_messages_queue
    for message_content in azure_messages:
        message = Message(message_content)

        try:
            device_client.send_message(message)
            print(f"Message sent: {message_content}")

        except Exception as ex:
            print("Error occurred while sending message:", ex)
            sys.exit()
                
        azure_messages_queue.pop(0)

    azure_messages = []

def analyze_img():
    global CONFIDENCE_THRESHOLD
    global FIRST_GOAL_COORDS
    global GOAL_1_HEIGHT
    global azure_messages_queue
    global pass_num
    global shot_num
    global goal_num
    global coords_ball_dict
    global coords_player_dict
    global is_running
    global video_cap
    global tracker
    global program_start
    global writer
    global model
    global device_client
    global first_frame
    global ball_coords_1
    global ball_coords_2
    global ball_coords_y
    global ball_coords_x
    global pitch_x_coord
    global pitch_y_coord
    global pass_cooldown
    global shot_cooldown

    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break
            
        line(frame, (FIRST_GOAL_COORDS[0], FIRST_GOAL_COORDS[1]), (FIRST_GOAL_COORDS[2], FIRST_GOAL_COORDS[3]), (255, 0, 0), FIRST_GOAL_COORDS[4])
        line(frame, (FIRST_GOAL_COORDS[0], FIRST_GOAL_COORDS[1]), (FIRST_GOAL_COORDS[0], FIRST_GOAL_COORDS[1] + GOAL_1_HEIGHT), (255, 0, 0), FIRST_GOAL_COORDS[4])
        line(frame, (FIRST_GOAL_COORDS[2], FIRST_GOAL_COORDS[3]), (FIRST_GOAL_COORDS[2], FIRST_GOAL_COORDS[3] + GOAL_1_HEIGHT), (255, 0, 0), FIRST_GOAL_COORDS[4])

        detections = model(frame)[0]

        results = []

        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            if int(data[5]) == 1:
                class_id = "Player"
            elif int(data[5]) == 0:
                class_id = "Ball"
            else:
                class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            rectangle(frame, (xmin, ymin), (xmax, ymax), DARK_BLUE, 2)
            rectangle(frame, (xmin, ymin - 20), (xmin + 80, ymin), DARK_BLUE, -1)
            putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            putText(frame, class_id, (xmin + 25, ymin - 8),
                        FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                
            base_coord_x = xmax - ((xmax- xmin) / 2)
            base_coord_y = ymax - ((ymax- ymin) / 2)
                
            pitch_x_coord = int(-0.019161996070921516*base_coord_x + 22.827119901896854)
            pitch_y_coord = int(0.028366479708419892*base_coord_y + 22.504367127807107)

            if class_id == "Ball":
                if first_frame == True:
                    first_frame = False
                    ball_coords_1.append(pitch_x_coord)
                    ball_coords_1.append(pitch_y_coord)

                    ball_coords_x.append(pitch_x_coord)
                    ball_coords_y.append(pitch_y_coord)

                    ball_coords_2.append(pitch_x_coord)
                    ball_coords_2.append(pitch_y_coord)
                else:
                    if ball_coords_2[0] - ball_coords_1[0] > 30 or ball_coords_2[1] - ball_coords_1[1] > 30:
                        if shot_cooldown <= 0:
                            shot_num += 1
                            shot_cooldown = 48
                            pass_cooldown = 48
                            print("Shot")
                    elif ball_coords_2[0] - ball_coords_1[0] > 10 or ball_coords_2[1] - ball_coords_1[1] > 10:
                        if pass_cooldown <= 0:
                            putText(frame, "Pass", (50, 500),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                            pass_num += 1
                            pass_cooldown = 48
                            print("Pass")                   
                    ball_coords_1 = ball_coords_2
                    ball_coords_2 = [pitch_x_coord, pitch_y_coord]
                    
                coords_ball_dict["x"] = pitch_x_coord
                coords_ball_dict["y"] = pitch_y_coord
                coords_ball_dict["passes"] = pass_num
                coords_ball_dict["goals"] = goal_num
                coords_ball_dict["shots"] = shot_num

                message_content = json.dumps(coords_ball_dict)
                azure_messages_queue.append(message_content)

            elif class_id == "Player":
                coords_player_dict["player_id"] = track_id
                coords_player_dict["x"] = pitch_x_coord
                coords_player_dict["y"] = pitch_y_coord
                coords_player_dict["passes"] = pass_num
                coords_player_dict["goals"] = goal_num
                coords_player_dict["shots"] = shot_num

                message_content = json.dumps(coords_player_dict)
                azure_messages_queue.append(message_content)
                coords_player_dict = {
                    "player_id": 0,
                    "x": 0,
                    "y": 0,
                    "passes": 0,
                    "shots": 0,
                    "goals": 0
                }

        pass_cooldown -= 1
        shot_cooldown -= 1

        end = datetime.datetime.now()
        print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        putText(frame, fps, (50, 50),
                    FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        cv2.imshow("Frame", frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord("q"):
            break

    program_end = datetime.datetime.now()
    program_time = (program_end - program_start).total_seconds()

    print("")
    print(f"Time to process the whole video: {program_time} seconds")
    is_running = False

def main():
    global FIRST_GOAL_COORDS
    global video_cap
    global tracker
    global program_start
    global writer
    global model
    global device_client
    global x_values
    global y_values
    global pitch_x_coord
    global pitch_y_coord

    slope = (FIRST_GOAL_COORDS[3] - FIRST_GOAL_COORDS[1]) / (FIRST_GOAL_COORDS[2] - FIRST_GOAL_COORDS[0])
    x_values = linspace(FIRST_GOAL_COORDS[0], FIRST_GOAL_COORDS[2], num=360)
    y_values = slope * (x_values - FIRST_GOAL_COORDS[0]) + FIRST_GOAL_COORDS[1]

    video_name = str(input("Video name: "))
    print("")

    try:
        device_client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        device_client.connect()
    except:
        sys.exit("Can't connect to the IotHub")    

    video_cap = cv2.VideoCapture(video_name)

    vd_name, file_ext = video_name.split(".")
    vd_name = vd_name.strip()
    writer = create_video_writer(video_cap, f"{vd_name}_output.mp4")

    model = YOLO("runs/best_weights/yolov8l/v7/best.pt")
    tracker = DeepSort(MAX_AGE)

    program_start = datetime.datetime.now()

    t1 = threading.Thread(target=analyze_img)
    t2 = threading.Thread(target=run_sample, args=(device_client,))
    t3 = threading.Thread(target=detect_goal)

    t1.daemon = True
    t2.daemon = True
    t3.daemon = True

    t1.start()
    t2.start()
    t3.start()

    while is_running == True or len(azure_messages) != 0:
        sleep(1)

    t1.join()
    t2.join()
    t3.join()

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()
    device_client.disconnect()
    device_client.shutdown()

if __name__ == "__main__":
    main()
