from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

camera = cv2.VideoCapture(0) 

def forFrame(frame_number, output_array, output_count):


    print("Frame Number : " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_27")
                                , frames_per_second=1, log_progress=True, per_frame_function = forFrame)
print(video_path)