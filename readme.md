
A toolbox for predicting clinical outcome of Alzheimer’s Diseases.
It will generally run with Edward on top of tensorflow. 

Main tools

Train_model: Load the training dataset, train the prediction model and save the model.
Test_model: Load the testing dataset, run the trained model on different testing subjects and save the results.
Idx: The subject ID we used from https://tadpole.grand-challenge.org/data/
Read_data: Read the raw dataset and convert to our input files.
test-run.ipynb: The juypter notebook file to run the train_model.py and test_model.py.
Y1: MMSE (0-30)
Y2: ADAS-COG(0-70)
Y3: normalized ventricle volume 
Y4: normalized hippocampus volume
Y5: CDRSB(0-17). 
X:  APOE4, gender, education, normalized ventricle volume and hippocampus volume at baseline visit

Requirements:
•	Tensorflow, Edward and all of their requirements (e.g. hyp5). Please install following packages to run our code: tensorflow: https://www.tensorflow.org Edward: http://edwardlib.org/
•	numpy, scipy, tqdm
Development:
Please contact yingying zhu, yz2377@cornell.edu for question related.

Papers:
If you use this code, please cite one of the following:
Yingying Zhu, Mert R. Sabuncu,
A Probabilistic Disease Progression Model for Predicting Future Clinical Outcome,
Medical Image Computing and Computer-Assisted Intervention (MICCAI) workshop, 2018.

