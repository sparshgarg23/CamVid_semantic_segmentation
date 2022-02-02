# CamVId_Bisenet
Train and test BiseNet architecure on CamVid to get an understanding  of new semantic segmentation architectures.
#Obtain camvid dataset from google drive https://drive.google.com/file/d/1KRRME_NtRG-iWOyLAb7gE-eA8fTeyzUR/view
#If you are interested in just running the demo,please mention the path to best_dice_loss.pth(it's currently stored in models)
#The architecture was trained on colab for 100 epochs with each epoch taking 2.3 minutes ,suggestions on improving the training process are welcome.
#In order to visualize logs please use the following commands
1.%load_ext tensorboard
2.%tensorboard --logdir runs/
#Epoch vs training loss
![epoch_loss_epoch_train](https://user-images.githubusercontent.com/34626942/152098616-463a704d-7e1c-4fb4-9df2-c39197f245df.jpg)
#Mean IOU
![epoch_miou val](https://user-images.githubusercontent.com/34626942/152098651-955263dd-c15f-4f8e-b14b-e77132867ca5.jpg)
#Epoch precision
![epoch_precision_val](https://user-images.githubusercontent.com/34626942/152098693-7cc10928-7a50-424b-85d0-0ad2c1ca0e97.jpg)

#Some results
#Demo picture
![exp](https://user-images.githubusercontent.com/34626942/152098739-6ec7a68d-5b68-41ca-b0bd-536a05a250e9.png)
#Result
![person_segmentation_100](https://user-images.githubusercontent.com/34626942/152098769-72b61ee4-4c30-4b78-a3fd-0b0d734cb7d0.png)
#Ground truth
![Seq05VD_f01560_L](https://user-images.githubusercontent.com/34626942/152099460-26313912-e98b-4f9d-ac75-f6958b9c0cf5.png)
#Future changes:experiment with more architectures ,train for longer duration and with batch size 16
