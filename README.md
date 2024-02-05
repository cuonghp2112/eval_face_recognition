# eval_face_recognition

### Download mediapipe detector tflite model
https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

### Download face recogintion model (backbone.pth or resnet50.onnx)
https://www.kaggle.com/datasets/cuongpham2112/test-1801

### Install requirements
```shell
pip install -r requirements.txt
```

### Run evaluation
Prepare test data dir 
```
├── images/
│ ├── person_0001/
│ │ ├── img_pr.bmp or img_pr.ppm.bz2
│ │ └──....
│ ├── person_0002
│ ├──....
```
- All images in the test data folder must have the same format.
- Currently, supported formats include BMP, JPG, PNG, and compressed PPM images in the .bz2 format (ppm.bz2)

Run the evaluation.py

```shell
python evaluation.py --test-dir path_to_test_data_dir \
                     --save-dir path_to_save_output_results \
                     --img-format "ppm.bz2" \
                     --face-detect path_to_face_detect_model \
                     --face-recog path_to_backbone.pth_or_onnx_model \
                     --device cpu
```
