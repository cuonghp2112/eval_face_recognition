import pandas as pd
import os, sys
from random import shuffle
import glob
import numpy as np
import shutil
from .utils import read_image_from_file, read_image_from_bz2
from .face_recog_pipeline import FaceRecogPipeline

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
            prediction = self.model.predict(image)[0]

            prediction_str = ",".join(str(x) for x in prediction)

            with open(csv_path, mode='w') as csv_file:
                csv_file.write(prediction_str)


    def distance(self, prediction_str1, prediction_str2):
        vector1 = np.array(prediction_str1.split(",")).astype(np.float32)
        vector2 = np.array(prediction_str2.split(",")).astype(np.float32)
        return 1 - np.dot(vector1,vector2.T)

    def get_folder_from_path(self, image_path):
        image_path = image_path.split("\\")[-3]
        return image_path

    def evaluate(self):

        print(f"Predicting Images...............................................")
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\Users_Image_Test\\'
        # a = self.testing_data_folder.split("\\")[-2]
        SPLITTED_TEST_DATA_DIR2 = self.splitted_test_dir
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\dataset_contrast2\\Users_Image_Train\\'
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\dataset_contrast\\Users_Image_Train\\'
        # SPLITTED_TEST_DATA_DIR2 = os.path.join(self.splitted_test_dir, a)
        # SPLITTED_TEST_DATA_DIR2 = 'splited_test_dir\\imgDir\\Users_Image_Train\\'

        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            original_dir = os.path.join(user_dir, "Original")
            prediction_dir = os.path.join(user_dir, "Prediction")

            for image in glob.glob(os.path.join(original_dir, f"*.{self.img_format}")):
                try:
                    self.predict_and_save(image)
                except Exception as e:
                    print(f"error image: {image}")
                    print(e)
                    

            for image in glob.glob(os.path.join(prediction_dir, f"*.{self.img_format}")):
                try:
                    self.predict_and_save(image)
                except Exception as e:
                    print(f"error image: {image}")
                    print(e)

        print(f"Done Predicting.................................................")
        
        original_values = []
        # Get all original values
        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            original_dir = os.path.join(user_dir, "Original")
            for image in glob.glob(os.path.join(original_dir, f"*.{self.img_format}")):
                try:
                    line = [image]
                    csv_path = image.replace(self.img_format, "csv")
                    with open(csv_path, mode='rb') as csv_file:
                        prediction = csv_file.readline().decode("utf-8")
                        if len(prediction) == 0:
                            raise Exception("Empty result: ", csv_path)
                        line.append(prediction)
                except Exception as e:
                    print(f"error image: {image}")
                    print(e)
                    continue
                    
                original_values.append(line)

        print(f"Calculating Distance............................................")

        for user_dir in glob.glob(os.path.join(SPLITTED_TEST_DATA_DIR2, "*")):
            prediction_dir = os.path.join(user_dir, "Prediction")

            for image in glob.glob(os.path.join(prediction_dir, f"*.{self.img_format}")):
                csv_path = image.replace(self.img_format, "csv")
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
                image_list = []
                distance_list = []

                for original_value in original_values:
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

            for image in glob.glob(os.path.join(prediction_dir, f"*.{self.img_format}")):
                total_image += 1
                csv_out_path = image.replace(f".{self.img_format}", "_out.csv")
                data = pd.read_csv(csv_out_path)
                max_res = -1
                index_min = 0
                gt_user = self.get_folder_from_path(image)
                predict_image_path = data.iloc[0][0]
                predict_user = self.get_folder_from_path(predict_image_path)
                try:
                    retval, corr = cv_phase_correlate(image, predict_image_path)
                    max_res = corr
                except:
                    corr = -1
                size_loop = data.shape[0]-1
                if ( data.iloc[0][1] < 0.02 ) and gt_user != predict_user:
                    with open("ReplaceFiles.txt", mode='a') as prob_file:
                        prob_file.write(f"{gt_user},{predict_user}:{data.iloc[0][1]} \n")
                    with open("RemoveFiles.txt", mode='a') as prob_file1:
                        prob_file1.write(f"{predict_image_path} \n")

                    print(f" PROBLEM $$$$$$ {image}, {predict_image_path}:{data.iloc[0][1]}")

                if gt_user == predict_user and data.iloc[0][1] == 0:
                    max_res = 0


                if ( ( corr > 0.4 or data.iloc[0][1] < 0.2) and data.iloc[0][1] > 0 ) and gt_user == predict_user:
                    size_loop = 0
                try:
                    if gt_user != predict_user or (gt_user == predict_user and data.iloc[0][1] == 0):
                        for i in range(size_loop):
                            predict_image_path = data.iloc[i][0]
                            pred_res = data.iloc[i][1]
                            retval, corr = cv_phase_correlate( image, predict_image_path )
                        #     predict_user = self.get_folder_from_path(predict_image_path)
                            max_distance = data.iloc[i][1]
                            if max_res < corr and pred_res > 0:
                                max_res = corr
                                index_min= i
                                if max_res > 0.4:
                                    break
                except:
                    print(f"i {i} data.size {data.size}")
                predict_image_path = data.iloc[index_min][0]
                predict_user = self.get_folder_from_path(predict_image_path)
                min_distance = data.iloc[index_min][1]
                corr = max_res
                    # gt_user = self.get_folder_from_path(gt_user)
                # predict_user = self.get_folder_from_path(predict_user)
                # predict_image_path = data.iloc[0][0]
                # predict_user = self.get_folder_from_path(predict_image_path)
                # min_distance = data.iloc[0][1]
                # predict_user = self.get_folder_from_path(predict_user)

                if gt_user == predict_user:
                    correct_image +=1
                    print(f"Process image: {image.replace(SPLITTED_TEST_DATA_DIR2, '')} - Grouth truth: {gt_user} - Predict: {predict_user}")
                    # print(f"Correct! Min Distance: {min_distance} ")
                    # print("-----------------------------")
                    print(f"Correct! Min Distance: {min_distance} retval {retval} index_min {index_min} corr {corr} ")
                    # min_distance = data.iloc[0][1]
                    [phashvalue, ahashvalue, whashvalue, dhashvalue] = compareHash(image, predict_image_path)
                    totRes = (phashvalue + ahashvalue + whashvalue+ dhashvalue) / 4
                    if(  min_distance > 0.65 and  corr < 0.31 ):
                        FRR += 1
                        # print(f"Process image: {image.replace(SPLITTED_TEST_DATA_DIR2, '')} - Grouth truth: {gt_user} - Predict: {predict_user}")
                        # print(f"Correct! Threshold: {threshold}, Min Distance: {min_distance} totRes: {totRes}")
                        # print(f"Correct! Threshold: {phashvalue} phashvalue, {ahashvalue} ahashvalue, whashvalue {whashvalue}, dhashvalue {dhashvalue} ")
                        # print(f"retval {retval} corr {corr} ")


                        print("-^^^^^^^--------  FRR ------^^^^^^------------")

                    if threshold < min_distance:
                        threshold = min_distance
                    # print(f"Correct! Threshold: {threshold}, Min Distance: {min_distance}")
                else:
                    if (min_distance > 0 ):
                        incorrect_image += 1
                        [phashvalue, ahashvalue, whashvalue, dhashvalue] = compareHash(image, predict_image_path)
                        totRes = ( phashvalue + ahashvalue + whashvalue + dhashvalue ) / 4
                        print(f"Process image: {image.replace(SPLITTED_TEST_DATA_DIR2, '')} - Grouth truth: {gt_user} - Predict: {predict_user}")
                        print(f"Incorrect! Threshold: {threshold}, Min Distance: {min_distance} totRes: {totRes}")
                        # print(f"Correct! Threshold: {phashvalue} phashvalue, {ahashvalue} ahashvalue, whashvalue {whashvalue}, dhashvalue {dhashvalue} ")
                        print(f"retval {retval} corr {corr} ")
                        print("++++++++++++++++++++++++++++++")
                        if ( min_distance < 0.65  and  corr > 0.31 ):

                            # print(f"Process image: {image.replace(SPLITTED_TEST_DATA_DIR2, '')} - Grouth truth: {gt_user} - Predict: {predict_user}")
                            FAR +=1
                            # print(f"Incorrect! Threshold: {threshold}, Min Distance: {min_distance} totRes: {totRes}")
                            # print("++++++++++++++++++++++++++++++")
                    else:
                        print(f"!!!!!NotFound: - {gt_user} - Predict: {predict_user} min_distance {min_distance}")
                        print("-----------------------------------")
                # print("-----------------------------")

        print(f"Correct: {correct_image}, Incorrect: {incorrect_image}, FRR: {FRR} FAR: {FAR} ({correct_image + incorrect_image} total_image {total_image})")
        print(f"Final: {correct_image/total_image * 100}")
        print("Max Threhold", round(max_threshold,3))
        print("Min Threhold", round(min_threshold,3))
        # print("Mean Threshold", round(stats.mean(threshold),3))
        # print("Median Threshold", round(stats.median(threshold),3))
        # print("Median 75 Threshold", round(np.percentile(threshold,75),3))
        # print("Median 95 Threshold", round(np.percentile(threshold,95),3))