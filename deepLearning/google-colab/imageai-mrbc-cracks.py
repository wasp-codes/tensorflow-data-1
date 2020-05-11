#to be used as a notebook
!pip3 install tensorflow-gpu==1.13.1

!pip3 install imageai --upgrade

!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My Drive

from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="mrbc-cracking")
trainer.setTrainConfig(object_names_array=["crack"], batch_size=4, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()