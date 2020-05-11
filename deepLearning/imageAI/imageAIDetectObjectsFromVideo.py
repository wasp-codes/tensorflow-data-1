from imageai.Detection import VideoObjectDetection
import os



execution_path = os.getcwd()

def forFrame(frame_number, output_array, output_count):


    print("Frame Number : " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")


video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel(detection_speed="fast")


video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic-mini.mp4"), save_detected_video=False ,  frames_per_second=20, per_frame_function = forFrame, minimum_percentage_probability=30)