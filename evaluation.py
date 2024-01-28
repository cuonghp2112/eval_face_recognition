import pandas as pd
import os, sys, re
from random import shuffle
import glob
import numpy as np
import mediapipe as mp
import shutil
from utils import read_image_from_file, read_image_from_bz2
from face_recog_pipeline import FaceRecogPipeline

def copy_images(image_paths, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for image_path in image_paths:
        shutil.copy(image_path, target_dir)
        

class Evaluate:
    def __init__(self, testing_data_folder=None,
                 splitted_test_dir=None,
                 img_format:str="bmp",
                 face_det_model_path=None,
                 face_recog_model_path=None,
                 device:str="cpu"):
        self.testing_data_folder = testing_data_folder
        self.splitted_test_dir = splitted_test_dir
        self.model = FaceRecogPipeline(face_det_model_path=face_det_model_path,
                                              face_recog_model_path=face_recog_model_path, device=device)
        self.img_format = img_format
        
        if not os.path.exists(self.splitted_test_dir):
            os.mkdir(self.splitted_test_dir)

        # List all user folders
        for user_dir in glob.glob(os.path.join(self.testing_data_folder, "*")):
            # Get the folder name
            user_name = user_dir.split('/')[-1]
            # Create new folder in self.splitted_test_dir
            splitted_user_dir = os.path.join(self.splitted_test_dir, user_name)
            if not os.path.exists(splitted_test_dir):
                os.makedirs(splitted_user_dir)

            # Get list of images in folder
            images = glob.glob(os.path.join(user_dir, "*"))
            if (len(images) > 1):
                # Shuffle the image list
                shuffle(images)

                # Split 50/50 for testing
                total = int(len(images) / 2)

                # Original images
                original_images = images[:total]
                copy_images(original_images, os.path.join(splitted_user_dir, "Original"))

                # Prediction images
                prediction_images = images[total:]
                copy_images(prediction_images, os.path.join(splitted_user_dir, "Prediction"))
                
    def load_image(self, image_path):
        if image_path.endswith(".bz2"):
            img_raw = read_image_from_bz2(image_path)
        else:
            img_raw = read_image_from_file(image_path)
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_raw)
        return img_mp

    def predict_and_save(self, image_path):
        csv_path = image_path.replace(self.img_format, "csv")
        if not os.path.exists(csv_path):

            image = self.load_image(image_path)
            prediction, head_pose = self.model.predict(image)
            prediction_str = ",".join(str(x) for x in prediction[0])
            csv_path = csv_path.replace(".csv",f"_{head_pose}.csv")

            with open(csv_path, mode='w') as csv_file:
                csv_file.write(prediction_str)


    def distance(self, prediction_str1, prediction_str2):
        vector1 = np.array(prediction_str1.split(",")).astype(np.float32)
        vector2 = np.array(prediction_str2.split(",")).astype(np.float32)
        return 1 - np.dot(vector1,vector2.T)

    def get_folder_from_path(self, image_path):
        if os.uname().sysname.lower() == "linux":
            image_path = image_path.split("/")[-3]
        else:
            image_path = image_path.split("\\")[-3]
        return image_path
    
    def pose_from_csv_filename(self, file_name):
        pose = re.search("(?<=_)(up|down|left|right|forward)(?=\..+$)",file_name)
        return pose.group(0) if pose is not None else None
    
    def image_from_saved_csv(self, file_name):
        image_name = re.sub("(?<=_)(up|down|left|right|forward)(?=\..+$)","",file_name)
        return image_name

    def evaluate(self):

        print(f"Predicting Images...............................................")
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\Users_Image_Test\\'
        # a = self.testing_data_folder.split("\\")[-2]
        SPLITTED_TEST_DATA_DIR2 = self.splitted_test_dir
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\dataset_contrast2\\Users_Image_Train\\'
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\dataset_contrast\\Users_Image_Train\\'
        # SPLITTED_TEST_DATA_DIR2 = os.path.join(self.splitted_test_dir, a)
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\imgDir\\Users_Image_Train\\'
        
        error_original = []
        error_prediction = []

        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            original_dir = os.path.join(user_dir, "Original")
            prediction_dir = os.path.join(user_dir, "Prediction")

            for image in glob.glob(os.path.join(original_dir, f"*.{self.img_format}")):
                try:
                    self.predict_and_save(image)
                except Exception as e:
                    print(f"error image: {image}")
                    error_original.append(image)
                    print(e)
                    

            for image in glob.glob(os.path.join(prediction_dir, f"*.{self.img_format}")):
                try:
                    self.predict_and_save(image)
                except Exception as e:
                    print(f"error image: {image}")
                    error_prediction.append(image)
                    print(e)
        print("error predict Original image")
        print(error_original)
        print("-"*30 + "\n")
        
        print("error predict Prediction image")
        print(error_prediction)
        print("-"*30 + "\n")

        print(f"Done Predicting.................................................")
        
        original_values = []
        # Get all original values
        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            original_dir = os.path.join(user_dir, "Original")
            for csv_path in glob.glob(os.path.join(original_dir, "*.csv")):
                try:
                    image_path = self.image_from_saved_csv(csv_path)
                    head_pose = self.pose_from_csv_filename(csv_path)
                    line = [image_path]
                    with open(csv_path, mode='rb') as csv_file:
                        prediction = csv_file.readline().decode("utf-8")
                        if len(prediction) == 0:
                            raise Exception("Empty result: ", csv_path)
                        line.append(prediction)
                        line.append(head_pose)
                except Exception as e:
                    print(f"error image: {image}")
                    print(e)
                    continue
                    
                original_values.append(line)

        print(f"Calculating Distance............................................")
        count = 0
        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            prediction_dir = os.path.join(user_dir, "Prediction")

            for csv_path in glob.glob(os.path.join(prediction_dir, "*.csv")):
#                 csv_path = image.replace(self.img_format, "csv")
                csv_out_path = csv_path.replace(".csv", "_out.csv")
                prediction = ""
                try:
                    with open(csv_path, mode='rb') as csv_file:
                        prediction = csv_file.readline().decode("utf-8")
                    if len(prediction) == 0:
                        raise Exception("Empty result: ", csv_path)
                except Exception as e:
                    print(f"error csv: {csv_path}")
                    print(e)
                    continue
                head_pose = self.pose_from_csv_filename(csv_path)
                image_list = []
                distance_list = []

                for original_value in original_values:
                    #  skip image has different head pose
                    if original_value[2] != head_pose:
                        continue
                    dist = self.distance(prediction, original_value[1])
                    image_list.append(original_value[0])
                    distance_list.append(dist)

                data = pd.DataFrame(
                    {'image': image_list,
                    'distance': distance_list
                    })
                data = data.sort_values('distance')
                data.to_csv(csv_out_path, index=False)


        
        total_image=0
        correct_image = 0
        incorrect_image = 0
        max_threshold = 0.0
        min_threshold = 100
        FRR= 0
        FAR=0
        threshold = []
        
        print(SPLITTED_TEST_DATA_DIR2)
        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            prediction_dir = os.path.join(user_dir, "Prediction")
            for csv_path in glob.glob(os.path.join(prediction_dir, "*.csv")):

                if csv_path.endswith("_out.csv"):
                    continue
                total_image += 1
                csv_out_path = csv_path.replace(".csv", "_out.csv")
                image_path = csv_path
                
                data = pd.read_csv(csv_out_path)
                index_min = 0
                gt_user = self.get_folder_from_path(image_path)
                predict_image_path = data.iloc[0]["image"]
                predict_user = self.get_folder_from_path(predict_image_path)

                if ( data.iloc[0]['distance'] < 0.02 ) and gt_user != predict_user:
                    with open("ReplaceFiles.txt", mode='a') as prob_file:
                        prob_file.write(f"{gt_user},{predict_user}:{data.iloc[0]['distance']} \n")
                    with open("RemoveFiles.txt", mode='a') as prob_file1:
                        prob_file1.write(f"{predict_image_path} \n")

                    print(f" PROBLEM $$$$$$ {image_path}, {predict_image_path}:{data.iloc[0]['distance']}")


                predict_image_path = data.iloc[index_min]['image']
                predict_user = self.get_folder_from_path(predict_image_path)
                min_distance = data.iloc[index_min]['distance']


                if gt_user == predict_user:
                    correct_image +=1
                    print(f"Process image: {image_path.replace(SPLITTED_TEST_DATA_DIR2, '')} - Grouth truth: {gt_user} - Predict: {predict_user}")
                    print(f"Correct! Min Distance: {min_distance} index_min {index_min}")
                    if(  min_distance > 0.3 ):
                        FRR += 1

                        print("-^^^^^^^--------  FRR ------^^^^^^------------")

                    if min_threshold > min_distance:
                        min_threshold = min_distance
                    
                    if max_threshold < min_distance:
                        max_threshold = min_distance
                    
                    threshold.append(min_distance)

                else:
                    if (min_distance > 0 ):
                        incorrect_image += 1
                        print(f"Process image: {image_path.replace(SPLITTED_TEST_DATA_DIR2, '')} - Grouth truth: {gt_user} - Predict: {predict_user}")
                        print(f"Incorrect! Min Distance: {min_distance}")

                        print("++++++++++++++++++++++++++++++")
                        if ( min_distance < 0.3 ):
                            FAR +=1
                    else:
                        print(f"!!!!!NotFound: - {gt_user} - Predict: {predict_user} min_distance {min_distance}")
                        print("-----------------------------------")
                # print("-----------------------------")

        print(f"Correct: {correct_image}, Incorrect: {incorrect_image}, FRR: {FRR} FAR: {FAR} ({correct_image + incorrect_image} total_image {total_image})")
        print(f"Final: {correct_image/total_image * 100}")
        print("Max Threhold", round(max_threshold,3))
        print("Min Threhold", round(min_threshold,3))
