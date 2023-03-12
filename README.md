# Master’s in Mathematics: Final Year Project 2018-2019 - Application of Discriminant Analysis for Predicting Medical Diagnosis of Cancer & Diabetes and Identification of Compounds in Herbal Plants 

## Project Objective

To test the Prediction Accuracy of various Hybrid versions of the KNN, LDA & QDA algorithms and compare them with their original versions

## Literature Survey

1. Halil Yigit had worked on ABC- based distance–weighted kNN algorithm [7].
2. Xuejun Ma, et.al [13] studied on a variant of K nearest neighbour quantile regression.
3. Some of the related works based on KNN and Discriminant Analysis can be found in [1, 8, 9, 10, 11, 12].  
4. The working of this project is based on the study of molecular descriptors for Drug/Non-Drug compounds extracted from medicinal plants. The molecular descriptors of these chemical compounds are identified by Dr. O. S. Deepa Gopakumar and Ani R. They have studied a lot on the classification of the compounds extracted from the medicinal plants [2, 3, 4, 5, 6].
5. Descriptions of the same can be found in [15, 16].
6. The machine learning approaches like classification of these compounds to drug compounds and non drug compounds are done by Kormaz, Selcuk, Gokmen Zararsiz, and Dincer Goksusluk [14].
7. Descriptions of machine learning approaches are also found in [17].

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

## Overview of KNN Classifier

1. This classifier does classification using instance based learning
2. KNN stands for k-nearest neighbours
3. This algorithm consists of two phases – training phase and testing phase
4. Training phase consists only of storing the feature vectors and class labels of the training sample
5. K is a user defined parameter which tells us how many neighbours to consider from the testing sample
6. The distances from each training point to the testing point are calculated
7. Identify the k-nearest neighbours
8. Use the class labels of these k-nearest neighbours to determine the class label for the testing sample
9. The testing sample is assigned the class label of the majority of the k-nearest neighbours

# Linear Discriminant Analysis

1. This is a method that helps us find a linear combination of features that splits two classes of objects from each other.
2. It is a generalization of Fischer’s Linear Discriminant, used in statistics, machine learning and pattern recognition.
3. The linear combination obtained as a result of LDA can be used as a linear classifier or for dimensionality reduction before classification.
4. It is closely related to Analysis of Variance(ANOVA), Regression Analysis, Principal Component Analysis(PCA) and Factor Analysis(FA), which also attempt to express a dependent variable in terms of independent variables.
5. This method is also known as Normal Discriminant Analysis(NDA) or Discriminant Function Analysis(DFA).

![alt text]()

# Quadratic Discriminant Analysis

1. This method is used to separate two or more classes of objects from one another.
2. Instead of a line in the case of LDA, here, we represent the boundary separating the classes as a quadric surface.
3. We can call this as a more generic version of the linear classifier.
4. It is used in machine learning and is closely related to LDA.
5. The measurements from each class are normally distributed like in LDA, but here, we do not assume that the covariance of each of the classes are equal.

![alt text]()

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

1. For all the three data sets used above, Diabetes, Cancer and Herbal Plants, the KNN LDA, KNN QDA, Condensed KNN LDA, Condensed KNN QDA, Fuzzy KNN LDA, Fuzzy KNN QDA, Constrained Fuzzy KNN LDA, Constrained Fuzzy KNN QDA, Rough Fuzzy KNN LDA and Rough Fuzzy KNN QDA showed a better performance than KNN, Condensed KNN, Constrained KNN and Rough Fuzzy KNN. 
2. From the cancer data set it is found that the accuracy performance of LDA and QDA with respect to Condensed, Fuzzy, Constrained Fuzzy and Rough Fuzzy were high. Also accuracy of QDA on Condensed Fuzzy, Constrained Fuzzy, and Rough Fuzzy over whelmed the accuracy with respect to LDA and hence can be concluded that the use of KNN QDA on Condensed Fuzzy, Constrained Fuzzy and Rough Fuzzy is preferable.  
3. The prediction for diabetes using various algorithms was around 77% and can be improved by using few boosting techniques or by increasing the number of samples. So the computer based decision support systems can be used based on Fuzzy KNN- LDA, Fuzzy KNN- QDA, KNN-LDA and KNN -QDA so as to reduce cost and errors in clinical trials. 
4. In the herbal plants data set, we find the LDA to be more effective than QDA with respect to ordinary KNN as well as with respect to Constrained Fuzzy. With respect to Fuzzy, Rough Fuzzy and Condensed Fuzzy, LDA and QDA are equally effective with higher accuracy in Rough Fuzzy than in Fuzzy and Condensed Fuzzy.

## References

1. Aiman Moldagulova, Rosnafisha Bte Sulaiman, Using KNN algorithm for classification of textual documents, IEEE Xplore October 2017.  
2. Ani R. and O.S. Deepa, Rotation forest ensemble algorithm for the classification of phyto-chemicals from the medicinal plants, Journal of chemical and pharmaceutical science, pp. 14-17, Special issue  4,  2016.  
3. Ani R., Jose J. ,Wilson M., Deepa O.S.: Modified rotation forest ensemble classifier for medical diagnosis in decision support systems Advances in Intelligent Systems and Computing,564, pp. 137-146. 
3. Ani R., Krishna S., Anju N., Sona A.M., Deepa O.S.: IoT based patient monitoring and diagnostic prediction tool using ensemble classifier, 2017 International Conference on Advances in Computing, Communications and Informatics, ICACCI 2017, 2017 January, pp. 1588-1593. 
5. Ani R. and Deepa O.S.: Rotation forest ensemble algorithm for the classification of phyto-chemicals from the medicinal plants, Journal of Chemical and Pharmaceutical Sciences,2016(SpecialIssue4), pp. 6-9 
6. Deepa O.S. and Ani R.: Expectation - Maximization algorithm for protein - Ligand complex of HFE gene, Journal of Chemical and Pharmaceutical Sciences, pp. 14-17, 2016(SpecialIssue4),  
7. Halil Yigit:  ABC – based distance - weighted kNN algorithm, Journal of Experimental and Theoretical Artificial Intelligence, Vol.27, issue 2, 2015.  
8. Liwen Huang and Lianta Su: Hierarchical Discriminant analysis and its application, Liwen  Huang and Lianta Su, Communication in Statistics – Theory and Methods, Vol 42, issue 11, 2013.  
9. P. Kakaivani, K.L. Shumuganathan: An improved K – nearest neighbour algorithm using   genetic algorithm for sentiment classification, IEEE Xplore, March 2015.  
10. P.T. Pepler, D.W Uys and D.G. Nel ; Discriminant analysis under the common principal  components model, Communication in statistics- simulation and computations, vol 46, issue 6, Feb 2017.  
11. Shweta Taneia, Charu Gupta, Kratika Goval, Dharna Gureia: An Enhanced K-Nearest Neighbour Algorithm Using Information Gain and clustering, IEEE Xplore April 2014.  
12. Wei-Yin Loh and Nutal Vanichesetakal: Journal of the American Statistical Association, Vol 83, issue 403, Mar 2012 Tree-Structured Classification via Generalized Discriminant Analysis.  
13. Xuejun Ma, Xiaogun He and Xiaokang Shi : A variant of K nearest neighbour quantile  regression, Journal of Applied Statistics, Vol 43,issue 3 ,2016. 
14. Kormaz, Selcuk, Gokmen Zararsiz, and Dincer Goksusluk. “Drug/nondrug classification using support vector machines with various feature selection strategies.” computer methods and programs in biomedicine 117.2(2014): 51-60.
15. Cano, Gaspar, et al. “Automatic selection of molecular descriptors using random forest: Application to Drug discovery.” Expert Systems with Applications 72(2017): 151-159.
16. Rodriguez, Juan Jose. Ludmila I. Kuncheva, and Carlos J. Alonso. ‘Rotation forest: A new classifier ensemble method.” IEEE transactions on pattern analysis and machine intelligence 28.10(2006): 1619-1630.
17. Lavecchia, Antonio. “Machine-learning approaches in Drug discovery: methods and applications.” Drug discovery today 20.3(2015): 318-331.

![image](https://user-images.githubusercontent.com/103128153/224525959-9ae123a1-5880-4b6a-b905-8276692535c9.png)



## Note

I had prepared the below slides for my Final Year Project Review in MSc Mathematics. These slides contain the detailed workings of each of the algorithms as well as their accuracies

<a href = "https://docs.google.com/presentation/d/10_1gMlXmtsN2yXt4Mf8IGrdLbQecxlNh/edit#slide=id.p1"> MSc Mathematics Final Year Project Review </a>

Further background behind this project is given in the thesis below which I submitted for my Degree in MSc Mathematics in PDF format.

<a href = "https://drive.google.com/drive/u/1/folders/1omvJKv7-dhtRXLOzdkoWOoOuNUG3nXsA"> HYBRID K-NEAREST NEIGHBOUR AND DISCRIMINANT ANALYSIS FOR PREDICTING MEDICAL DIAGNOSIS IN DECISION SUPPORT SYSTEM </a>
