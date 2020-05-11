from imageai.Prediction.Custom import ModelTraining

trainer = ModelTraining()
trainer.setModelTypeAsResNet()
trainer.setDataDirectory("fruits")
trainer.trainModel(num_objects=2, num_experiments=20, enhance_data=True, save_full_model=True, batch_size=8, show_network_summary=True, transfer_from_model="resnet50_weights_tf_dim_ordering_tf_kernels.h5", initial_num_objects=1000, transfer_with_full_training=True)
