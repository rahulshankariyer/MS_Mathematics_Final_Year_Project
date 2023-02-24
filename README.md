# Master’s in Mathematics: Final Year Project 2018-2019 - Application of Discriminant Analysis for Predicting Medical Diagnosis of Cancer & Diabetes and Identification of Compounds in Herbal Plants 

## Project Objective

To test the Prediction Accuracy of various Hybrid versions of the KNN, LDA & QDA algorithms and compare them with their original versions

## Data Source

The data used for testing the Prediction Accuracy was taken from the UCI Repository

## Data Used

In this project, the following datasets were used for Prediction of Medical Diagnosis:

1. Diabetes Dataset
2. Cancer Dataset
3. Herbal Plants Dataset

## About Each Dataset

### a. Diabetes Dataset

The Diabetes Dataset contains a binary attribute that denotes whether the patient has Diabetes or not, which is based on 20 Clinical Attributes.

### b. Cancer Dataset

The Cancer Dataset contains a binary attribute that denotes whether the patient has Cancer or not, which is based on 9 Clinical Attributes.

### c. Herbal Plants Dataset

The Herbal Plants Dataset contains 23 Drug Properties along with an attribute that denotes which Chemical Compound is available in the plant. The Diabetes & Cancer Datasets serve as the bench mark for this classification.

## Data Transformations

### a. Diabetes Dataset

The Diabetes Dataset was divided into 10 different partitions where one partition is known as a Training Dataset and the other 9 partitions are called Testing Datasets. 

### b. Cancer Dataset

The Cancer Dataset was divided in the same manner as the Diabetes Dataset and the classification followed the same procedure too. 

### c. Herbal Plants Dataset

The original dataset contained information for 143 Herbal Plants, so it was divided into two partitions where one partition containing the information of 71 Herbal Plants was taken as the Training Dataset and the other partition containing the information of the other 72 Herbal Plants was taken as Testing Dataset.

## Tools Used

1. Excel
2. MATLAB

## Analysis

Using MATLAB, the below hybrid versions of the KNN and Discriminant Analysis algorithms (listed below) were applied on our Testing Datasets

1. KNN (Original Algorithm)
2. LDA (Original Algorithm)
3. QDA (Original Algorithm)
4. KNN LDA (Hybrid Algorithm)
5. KNN QDA (Hybrid Algorithm)
6. FUZZY KNN (Original Algorithm)
7. FUZZY KNN LDA (Hybrid Algorithm)
8. FUZZY KNN QDA (Hybrid Algorithm)
9. ROUGH FUZZY KNN (Original Algorithm)
10. ROUGH FUZZY KNN LDA (Hybrid Algorithm)
11. ROUGH FUZZY KNN QDA (Hybrid Algorithm)
12. CONDENSED KNN (Original Algorithm)
13. CONDENSED KNN LDA (Hybrid Algorithm)
14. CONDENSED KNN QDA (Hybrid Algorithm)
15. CONSTRAINED KNN (Original Algorithm)
16. CONSTRAINED KNN LDA (Hybrid Algorithm)
17. CONSTRAINED KNN QDA (Hybrid Algorithm)

Predictions using each of the above were crried out and their accuracy was also computed. The accuracy is computed by comparing the predicted values with the actual values in the partitions taken as our Training Datasets.

### Accuracy On Diabetes Dataset

| Algorithm | Correctly Classified | Incorrectly Classified | Accuracy % |
| --- | --- | --- | --- |
| KNN | 76 | 24 | 76% |
| KNN LDA | 77 | 23 | 77% |
| KNN QDA | 77 | 23 | 77% |
| Fuzzy KNN | 76 | 24 | 76% |
| Fuzzy KNN LDA | 77 | 23 | 77% |
| Fuzzy KNN QDA | 77 | 23 | 77% |
| Rough Fuzzy KNN | 76 | 24 | 76% |
| Rough Fuzzy KNN LDA | 77 | 23 | 77% |
| Rough Fuzzy KNN QDA | 77 | 23 | 77% |
| Condensed KNN | 76 | 24 | 76% |
| Condensed KNN LDA | 77 | 23 | 77% |
| Condensed KNN QDA | 77 | 23 | 77% |
| Constrained KNN | 74 | 26 | 74% |
| Constrained KNN LDA | 77 | 23 | 77% |
| Constrained KNN QDA | 76 | 24 | 76% |

### Accuracy On Cancer Dataset

| Algorithm | Correctly Classified | Incorrectly Classified | Accuracy % |
| --- | --- | --- | --- |
| KNN | 96 | 4 | 96% |
| KNN LDA | 97 | 3 | 97% |
| KNN QDA | 97 | 3 | 97% |
| Fuzzy KNN | 97 | 3 | 97% |
| Fuzzy KNN LDA | 99 | 1 | 99% |
| Fuzzy KNN QDA | 99 | 1 | 99% |
| Rough Fuzzy KNN | 76 | 24 | 76% |
| Rough Fuzzy KNN LDA | 77 | 23 | 77% |
| Rough Fuzzy KNN QDA | 77 | 23 | 77% |
| Condensed KNN | 90 | 10 | 90% |
| Condensed KNN LDA | 92 | 8 | 92% |
| Condensed KNN QDA | 99 | 1 | 99% |
| Constrained KNN | 90 | 10 | 90% |
| Constrained KNN LDA | 92 | 8 | 92% |
| Constrained KNN QDA | 99 | 1 | 99% |

### Accuracy On Herbal Plants Dataset

| Algorithm | Correctly Classified | Incorrectly Classified | Accuracy % |
| --- | --- | --- | --- |
| KNN | 54 | 18 | 75% |
| KNN LDA | 69 | 3 | 95.83% |
| KNN QDA | 62 | 10 | 86.11% |
| Fuzzy KNN | 54 | 18 | 75% |
| Fuzzy KNN LDA | 68 | 4 | 94.4% |
| Fuzzy KNN QDA | 68 | 4 | 94.4% |
| Rough Fuzzy KNN | 54 | 18 | 75% |
| Rough Fuzzy KNN LDA | 69 | 3 | 95.83% |
| Rough Fuzzy KNN QDA | 69 | 3 | 95.83% |
| Condensed KNN | 54 | 18 | 75% |
| Condensed KNN LDA | 68 | 4 | 94.4% |
| Condensed KNN QDA | 68 | 4 | 94.4% |
| Constrained KNN | 54 | 18 | 75% |
| Constrained KNN LDA | 69 | 3 | 95.83% |
| Constrained KNN QDA | 62 | 10 | 86.11% |

In the below links, you can find the MATLAB codes for some of the algorithms as well as the code computing their accuracies

1. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/KNN.m"> KNN </a>
2. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/KNNAccuracy.m"> KNN Accuracy </a>
3. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/KNNLDA.m"> KNN LDA </a>
4. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/KNNLDAAccuracy.m"> KNN LDA Accuracy </a>
5. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/KNNQDA.m"> KNN QDA </a>
6. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/KNNQDAAccuracy.m"> KNN QDA Accuracy </a>
7. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/FuzzyKNN.m"> Fuzzy KNN </a>
8. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/FuzzyKNNAccuracy.m"> Fuzzy KNN Accuracy </a>
9. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/FuzzyLDA.m"> Fuzzy KNN LDA </a>
10. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/FuzzyLDAAccuracy.m"> Fuzzy KNN LDA Accuracy </a>
11. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/FuzzyQDA.m"> Fuzzy KNN QDA </a>
12. <a href = "https://github.com/rahulshankariyer/Portfolio/blob/main/Final%20Year%20Project%202018-2019/MATLAB/FuzzyQDAAccuracy.m"> Fuzzy KNN QDA Accuracy </a>

## Conclusion

For all the three data sets used above, the Hybrid Algorithms showed a better performance their respective Original Algorithms.

## Note

I had prepared the below slides for my Final Year Project Review in MSc Mathematics. These slides contain the detailed workings of each of the algorithms as well as their accuracies

<a href = "https://docs.google.com/presentation/d/10_1gMlXmtsN2yXt4Mf8IGrdLbQecxlNh/edit#slide=id.p1"> MSc Mathematics Final Year Project Review </a>

Further background behind this project is given in the thesis below which I submitted for my Degree in MSc Mathematics in PDF format.

<a href = "https://drive.google.com/drive/u/1/folders/1omvJKv7-dhtRXLOzdkoWOoOuNUG3nXsA"> HYBRID K-NEAREST NEIGHBOUR AND DISCRIMINANT ANALYSIS FOR PREDICTING MEDICAL DIAGNOSIS IN DECISION SUPPORT SYSTEM </a>
