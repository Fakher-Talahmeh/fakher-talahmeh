import cv2
import datetime
import imutils
import numpy as np

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

team_colors = {
    "Real Madrid": [(0, 120, 70), (10, 255, 255)],       # Range for red color
    "Barcelona": [(60, 100, 100), (70, 255, 255)],       # Range for blue color
    # Add more teams and their color ranges here
}
def main():
    cap = cv2.VideoCapture('teamm.mp4')


    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=700)
        (H, W) = frame.shape[:2]
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detector.setInput(blob)
        person_detections = detector.forward()
        for color_range in team_colors.values():
            lower_color = np.array(color_range[0])
            upper_color = np.array(color_range[1])
            color_mask = cv2.inRange(hsv_frame, lower_color, upper_color)
            mask = cv2.bitwise_or(mask, color_mask)
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            for contour in contours:
            # Filter out small contours (noise)
                if cv2.contourArea(contour) > 1000:
                # Get the bounding box of the contour
                    x, y, w, h = cv2.boundingRect(contour)
                # Extract the uniform color within the bounding box
                    player_color = hsv_frame[y + h // 2, x + w // 2]
                
                    player_color = cv2.cvtColor(np.uint8([[player_color]]), cv2.COLOR_HSV2RGB)
                
                # Determine the team based on the color range
                    pt = None
                    for team, color_range in team_colors.items():
                        lower_color = np.array(color_range[0], dtype=np.uint8)
                        upper_color = np.array(color_range[1], dtype=np.uint8)
                
                        if np.all(cv2.inRange(player_color, lower_color, upper_color)):
                            pt = team
                            break
                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    print(pt)
                    cv2.putText(frame, pt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()