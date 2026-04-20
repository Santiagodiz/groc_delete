# GROC_CICMalMem2022
Practica de la asignatura GROC 21/22, sobre generación de un modelo de ML usando SciKit-Learn con el dataset CICMalMem2022

Based on Charanpal Dhanjal work https://gist.github.com/charanpald/c216800e25480ee838e8

The dataset used is the CICMalMem2022 dataset https://www.unb.ca/cic/datasets/malmem-2022.html, a columns has been added to the rigth with the type of malware (Benign -no malware-, Ransomware, Spyware, Trojan).

To download dataset:
1) Go to https://www.unb.ca/cic/datasets/malmem-2022.html, navigate to the bottom of the page, click "Download the dataset".
   Type your data for recording purposes, you will be redirected to the dataset, download the .csv file. You may check the integrity of the file with the.md5 file. It occuoies about 16Mb, for the purpose of the practice, a randomly 10%-20% lines will suffice.
2) Run the code with
   python generateModel.py
3) Play with other classifiers, commenting RandomForest creation line
   (around line 54) and uncomment DecisionTreeClassifier.
