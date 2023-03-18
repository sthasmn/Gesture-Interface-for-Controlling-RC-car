import cv2
import mediapipe as mp
import time
import pandas as pd
import csv
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import SendToRaspPi as rp#SendToRaspi.py ファイルを同フォルダに保存する
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# For static images:
decide = True
count = 0
ptime = 0
close_check = False
open_check = False



def open_check_by_distance(keypoints, center):
    def thumb_open_check(keypoints, center):
        d4 = np.sqrt(np.square(keypoints[4][0] - center[0]) + np.square(keypoints[4][1] - center[1]))
        d3 = np.sqrt(np.square(keypoints[3][0] - center[0]) + np.square(keypoints[3][1] - center[1]))
        if d4 > d3:
            return True
        else:
            return False
    def index_open_check(keypoints, center):
        d5 = np.sqrt(np.square(keypoints[5][0] - center[0]) + np.square(keypoints[5][1] - center[1]))
        d6 = np.sqrt(np.square(keypoints[6][0] - center[0]) + np.square(keypoints[6][1] - center[1]))
        d7 = np.sqrt(np.square(keypoints[7][0] - center[0]) + np.square(keypoints[7][1] - center[1]))
        d8 = np.sqrt(np.square(keypoints[8][0] - center[0]) + np.square(keypoints[8][1] - center[1]))
        if d8 > d7 > d6 > d5:
            return True
        else:
            return False
    def middle_open_check(keypoints, center):
        d9 = np.sqrt(np.square(keypoints[9][0] - center[0]) + np.square(keypoints[9][1] - center[1]))
        d10 = np.sqrt(np.square(keypoints[10][0] - center[0]) + np.square(keypoints[10][1] - center[1]))
        d11 = np.sqrt(np.square(keypoints[11][0] - center[0]) + np.square(keypoints[11][1] - center[1]))
        d12 = np.sqrt(np.square(keypoints[12][0] - center[0]) + np.square(keypoints[12][1] - center[1]))
        if d12 > d11 > d10 > d9:
            return True
        else:
            return False
    def ring_open_check(keypoints, center):
        d13 = np.sqrt(np.square(keypoints[13][0] - center[0]) + np.square(keypoints[13][1] - center[1]))
        d14 = np.sqrt(np.square(keypoints[14][0] - center[0]) + np.square(keypoints[14][1] - center[1]))
        d15 = np.sqrt(np.square(keypoints[15][0] - center[0]) + np.square(keypoints[15][1] - center[1]))
        d16 = np.sqrt(np.square(keypoints[16][0] - center[0]) + np.square(keypoints[16][1] - center[1]))
        if d16 > d15 > d14 > d13:
            return True
        else:
            return False
    def pinky_open_check(keypoints, center):
        d17 = np.sqrt(np.square(keypoints[17][0] - center[0]) + np.square(keypoints[17][1] - center[1]))
        d18 = np.sqrt(np.square(keypoints[18][0] - center[0]) + np.square(keypoints[18][1] - center[1]))
        d19 = np.sqrt(np.square(keypoints[19][0] - center[0]) + np.square(keypoints[19][1] - center[1]))
        d20 = np.sqrt(np.square(keypoints[20][0] - center[0]) + np.square(keypoints[20][1] - center[1]))
        if d20 > d19 > d18 > d17:
            return True
        else:
            return False
    thumb = thumb_open_check(keypoints, center)
    index = index_open_check(keypoints, center)
    middle = middle_open_check(keypoints, center)
    ring = ring_open_check(keypoints, center)
    pinky = pinky_open_check(keypoints, center)
    if thumb == True and index == True and middle == True and ring == True and pinky == True:
        return True
    else:
        return False

def close_check_by_distance(keypoints, center): #tested OK
   d3 = np.sqrt(np.square(keypoints[3][0] - center[0]) + np.square(keypoints[3][1] - center[1]))
   d4 = np.sqrt(np.square(keypoints[4][0] - center[0]) + np.square(keypoints[4][1] - center[1]))
   d5 = np.sqrt(np.square(keypoints[5][0] - keypoints[0][0]) + np.square(keypoints[5][1] - keypoints[0][1]))
   d8 = np.sqrt(np.square(keypoints[8][0] - keypoints[0][0]) + np.square(keypoints[8][1] - keypoints[0][1]))
   d9 = np.sqrt(np.square(keypoints[9][0] - keypoints[0][0]) + np.square(keypoints[9][1] - keypoints[0][1]))
   d12 = np.sqrt(np.square(keypoints[12][0] - keypoints[0][0]) + np.square(keypoints[12][1] - keypoints[0][1]))
   d13 = np.sqrt(np.square(keypoints[13][0] - keypoints[0][0]) + np.square(keypoints[13][1] - keypoints[0][1]))
   d16 = np.sqrt(np.square(keypoints[16][0] - keypoints[0][0]) + np.square(keypoints[16][1] - keypoints[0][1]))
   d17 = np.sqrt(np.square(keypoints[17][0] - keypoints[0][0]) + np.square(keypoints[17][1] - keypoints[0][1]))
   d20 = np.sqrt(np.square(keypoints[20][0] - keypoints[0][0]) + np.square(keypoints[20][1] - keypoints[0][1]))

   if d8 < d5 and d12 < d9 and d16 < d13 and d20 < d17 and d4 < d3:
       return True
   else:
       return False

def take_coordinates(coordinates):
  #print(coordinates) 20 xyz
  if coordinates == None:
    return 0
  keypoints = []
  for data_point in coordinates:
    xyz_datapoints = data_point.landmark
    for xyz in xyz_datapoints:
      X_value = round(xyz.x*10000, 2)
      Y_value = round(xyz.y*10000, 2)
      Z_value = round(xyz.z, 3)
      xy = [X_value,Y_value, Z_value]
      keypoints.append(xy)
    #print("Depth情報:\n 手首:{} 中指先:{} 薬指先:{}".format(keypoints[0][2], keypoints[12][2], keypoints[16][2]))
  return keypoints

def centroid_palm(keypoints): #calculation not correct. Do it again
    if keypoints == 0:
        return 0
    x_bar = (keypoints[0][0] + keypoints[9][0])/2
    x_bar = round(x_bar, 2)
    y_bar = (keypoints[0][1] + keypoints[9][1])/2
    y_bar = round(y_bar, 2)
    return x_bar, y_bar

def distance_center_to_tip(keypoints, center):
    if keypoints == 0:
        return 0
    d1 = np.sqrt(np.square(keypoints[4][0] - center[0]) + np.square(keypoints[4][1] - center[1]))
    d2 = np.sqrt(np.square(keypoints[8][0] - center[0]) + np.square(keypoints[8][1] - center[1]))
    d3 = np.sqrt(np.square(keypoints[12][0] - center[0]) + np.square(keypoints[12][1] - center[1]))
    d4 = np.sqrt(np.square(keypoints[16][0] - center[0]) + np.square(keypoints[14][1] - center[1]))
    d5 = np.sqrt(np.square(keypoints[20][0] - center[0]) + np.square(keypoints[20][1] - center[1]))
    return round(d1, 1), round(d2, 1), round(d3, 1), round(d4, 1), round(d5, 1)

def get_angle(keypoints, center):
    #(x',y')=(x, max-y)
    if keypoints == 0:
        return 0

    center = list(center)
    wrist = list(keypoints)
    wrist[1] = 10000-wrist[1] # y' = max - y
    center[1] = 10000-center[1] # y' = max - y
    Y = center[1]-wrist[1]
    X = center[0]-wrist[0]
    try:
        m = Y/X
    except ZeroDivisionError:
        m = 0
    angle = np.arctan(m)*180/(np.pi)
    if X > 0 and Y < 0:
        angle = angle + 360
    elif X < 0 and Y > 0:
        angle = angle + 180
    elif X < 0 and Y < 0:
        angle = angle + 180
    return round(angle, 1)

def motor(value):#モーター制御のために必要な値にmappingを行う。モーターによって値が変わる
    leftMin = -0.3
    leftMax = 0.1
    rightMin = 6.5
    rightMax = 5.5

    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def calculate(keypoints):
    global decide
    global close_check
    global open_check

    if keypoints == 0:
        rp.send(f"{0}, {90}")
        return 0
    center = centroid_palm(keypoints) #ひらの中心を求める
    distance_center_to_tip_all = distance_center_to_tip(keypoints, center)
    angle = get_angle(keypoints[0], center)

    motor_value = round(motor(keypoints[12][2]), 1)#ラジコン車制御するためのPWM値を検出

    open_check = open_check_by_distance(keypoints, center)
    #print(motor_value)
    if open_check == True:
        rp.send(f"{motor_value}, {int(angle)}")#パソコンからRaspberry Piへモーターとサーボーの制御をするために送信を行う。
        print(f"sending{motor_value}, {angle}")
    elif close_check:
        rp.send(f"{6.0}, {90}")
    return angle# Test



IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.

    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )

      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

# For webcam input:
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
pTime = 0
with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    cv2.putText(image, f'FPS: {int(fps)}', (800, 720), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    keypoints = take_coordinates(results.multi_hand_landmarks)
    if keypoints != 0:
        place = (int((keypoints[12][0]) / 10), int((keypoints[12][1]) / 15))
        cv2.putText(image, f'{float(get_angle(keypoints[0], centroid_palm(keypoints)))}', place, cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
    if keypoints == 0:
        place = (200, 200)
    calculate(keypoints) #この関数が全ての数値を計算してる


    #cv2.putText(image, f'{float(get_angle(keypoints, centroid_palm(keypoints)))}', place, cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    #time.sleep(1)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()