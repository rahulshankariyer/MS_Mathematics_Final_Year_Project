# HYBRID K-NEAREST NEIGHBOUR AND DISCRIMINANT ANALYSIS FOR PREDICTING MEDICAL DIAGNOSIS IN DECISION SUPPORT SYSTEM - An Oversight

## Project Objective

To test the Prediction Accuracy of various Hybrid versions of the KNN, LDA & QDA algorithms and compare them with their original versions

## Data Source

The data used for testing the Prediction Accuracy was taken from the University of California Irvine Repository

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

The Herbal Plants Dataset contains 23 Drug Properties along with an attribute that denotes which Chemical Compound is available in the plants. 

## Data Transformations

### a. Diabetes Dataset

The Diabetes Dataset was divided into 10 different partitions where one partition is known as a Training Dataset and the other 9 partitions are called Testing Datasets. 

### b. Cancer Dataset

The Cancer Dataset was divided in the same manner as the Diabetes Dataset and the classification followed the same procedure. 

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

### Overview of KNN Classifier

1. This classifier does classification using instance based learning
2. KNN stands for k-nearest neighbours
3. This algorithm consists of two phases – training phase and testing phase
4. Training phase consists only of storing the feature vectors and class labels of the training sample
5. K is a user defined parameter which tells us how many neighbours to consider from the testing sample
6. The distances from each training point to the testing point are calculated
7. Identify the k-nearest neighbours
8. Use the class labels of these k-nearest neighbours to determine the class label for the testing sample
9. The testing sample is assigned the class label of the majority of the k-nearest neighbours

![alt text](https://raw.githubusercontent.com/rahulshankariyer/MS_Mathematics_Final_Year_Project/main/KNN.png)

### Linear Discriminant Analysis

1. This is a method that helps us find a linear combination of features that splits two classes of objects from each other.
2. It is a generalization of Fischer’s Linear Discriminant, used in statistics, machine learning and pattern recognition.
3. The linear combination obtained as a result of LDA can be used as a linear classifier or for dimensionality reduction before classification.
4. It is closely related to Analysis of Variance(ANOVA), Regression Analysis, Principal Component Analysis(PCA) and Factor Analysis(FA), which also attempt to express a dependent variable in terms of independent variables.
5. This method is also known as Normal Discriminant Analysis(NDA) or Discriminant Function Analysis(DFA).

![alt text](https://raw.githubusercontent.com/rahulshankariyer/MS_Mathematics_Final_Year_Project/main/LDA.png)

### Quadratic Discriminant Analysis

1. This method is used to separate two or more classes of objects from one another.
2. Instead of a line in the case of LDA, here, we represent the boundary separating the classes as a quadric surface.
3. We can call this as a more generic version of the linear classifier.
4. It is used in machine learning and is closely related to LDA.
5. The measurements from each class are normally distributed like in LDA, but here, we do not assume that the covariance of each of the classes are equal.

![alt text](https://raw.githubusercontent.com/rahulshankariyer/MS_Mathematics_Final_Year_Project/main/QDA.png)

### When to use LDA and QDA

1. LDA tends to be a better bet than QDA if there are relatively few training observations and so has substantially lower variance.
2. QDA is recommended if the training set is very large, so that the variance of the classifier is not a major concern.
3. QDA has more predictability power than LDA but it needs to estimate the covariance matrix for each classes. 

### Fuzzy KNN Algorithm

1. Take m sample vectors with n parameters, say, x1,x2,…,xn which can be assigned to any one of  different classes, say, C1,C2,…,Cn. These are considered as training data sets. 
2. For each of training and testing data the distance is calculated  by taking the sum of the squares of the distance between each parameter of the training and  testing data, i.e., given two vectors x1,x2,…,xn and y1,y2,…,yn.The distance is given by (x1-y1)2+(x2-y2)2+…+(xn-yn)2.  
3. Take the k training data sets with the least values of distance from the training data set.
4. From the k training data sets obtained above, count how many of these belong to each class.  
5. For each class, create a Boolean vector which assigns a value between 0 and 1.
6. For each class, calculate the value Ʃ(no of rows in each matrix*Covariance Matrix)/(distance of each training data set from the testing data set). These values are considered as XC1,XC2,…,XCn.  
7. Let Ʃ1/(distance of each training data set from the testing data set).  For each class, calculate the value Z=X/Y. Let these values be ZC1,ZC2,…,ZCn.
8. Assign the class with the highest Z-value to the testing data set. 
9. If two of the classes both share the highest Z-value, then find the closest of the k neighbours belonging to one of these two classes and assign that class to the testing data set.  
 
### Fuzzy KNN-LDA Algorithm

1. Take m sample vectors with n parameters, say, x1,x2,…,xn which can be assigned to any one of  different classes, say, c1,c2,…,cn. These are considered as training data sets. 
2. Take another sample vector, say k, with the same n parameters. We shall call this a testing data set. 
3. For each of training and testing data the distance is calculated  by taking the sum of the squares of the distance between each parameter of the training and  testing data, i.e., given two vectors x1,x2,…,xn and y1,y2,…,yn.The distance is given by (x1-y1)2+(x2-y2)2+…+(xn-yn)2.  
4. Take the k training data sets with the least values of distance from the training data set.
5. From the k training data sets obtained above, count how many of these belong to each class.  
6. For each class, store the belonging training data sets in a matrix, where each row of the matrix is a training data set. Call these as class matrices, say C1 and C2.
7. Calculate the average data set for the training data set in each matrix, say avgC1 and avgC2 for the distance vectors in both matrices, say avg.  
8. Multiply the distance vectors of each of the distance matrices of each class by their membership in that class. We shall call these matrices as d1 and d2.
9. For each of the class matrices, create new matrices, say C01 and C02, which are obtained by calculating the distance vector between each row of the class matrices and the average vector, avg.  
10. For each class we can calculate the covariance matrix using the formula (G*G’)/(number of rows in G) for any given matrix G. In this case, we shall apply the formula to the matrices C01 and C02 and call the resulting covariance matrices CovC1 and CovC2.  
11. Calculate the pooled covariance matrix, say Cov, which is given by the formula Ʃ(no of rows in each distance matrix*Covariance Matrix)/(no of rows in each distance matrix). Apply this formula to distance matrices d1 and d2 and covariance matrices CovC1 and CovC2.
12. Calculate the linear model coefficient vector, which is in this case, β=(avgC1-avgC2)/(Cov).
13. Let X=(K-((avgC1-avgC2)/2)) β’ and Y=-log((number of rows in C1)/(number of rows in C2)).
14. If X>Y, then assign the class c1 to the testing data set. Otherwise, assign the class c2.  

### Fuzzy KNN-QDA Algorithm

1. Take m sample vectors with n parameters, say, x1,x2,…,xn which can be assigned to any one of  different classes, say, c1,c2,…,cn. These are considered as training data sets. 
2. Take another sample vector, say k, with the same n parameters. We shall call this a testing data set. 
3. For each of training and testing data the distance is calculated  by taking the sum of the squares of the distance between each parameter of the training and  testing data, i.e., given two vectors x1,x2,…,xn and y1,y2,…,yn.The distance is given by (x1-y1)2+(x2-y2)2+…+(xn-yn)2.  
4. Take the k training data sets with the least values of distance from the training data set.
5. From the k training data sets obtained above, count how many of these belong to each class.  
6. For each class, store the belonging training data sets in a matrix, where each row of the matrix is a training data set. Call these as class matrices, say C1,C2,…,Cn.
7. Calculate the average data set for the training data set in each matrix, say avgC1, avgC2,…,avgCn for the distance vectors in both matrices, say avg.  
8. Multiply the distance vectors of each of the distance matrices of each class by their membership in that class. We shall call these matrices as d1,d2,…,dn.
9. For each of the class matrices, create new matrices, say C01,C02,…,C0n which are obtained by calculating the distance vector between each row of the class matrices and the average vector, avg.
10. For each class we can calculate the covariance matrix using the formula (G*G’)/(number of rows in G) for any given matrix G. In this case, we shall apply the formula to the matrices C01,C02,…,C0n and call the resulting covariance matrices CovC1,CovC2,…,CovCn.  
11. Calculate the pooled covariance matrix, say Cov, which is given by the formula Ʃ(no of rows in each distance matrix*Covariance Matrix)/(no of rows in each distance matrix).
12. Apply this formula to distance matrices d1,d2,…,dn and covariance matrices CovC1,CovC2,…,CovCn.
13. For  each  class,  calculate  ZC1=(-0.5*((1-(avgC1/CovC1))*(-avgC1)’-log(det(CovC1)))+log((number of rows in the distance matrix of (C1/k))
14. Apply this same formula to classes c1,c2,…,cn.  
15. Compare the values calculated in step 11.  
16. Assign the testing data set with the class having the greatest value calculated in step 11.  

### Rough KNN Algorithm

1. This algorithm is based on calculating ownership of each class towards the testing data set. 
2. The class with the highest ownership towards the testing data set is assigned to the testing data set.
3. We can perform two variants of this algorithm – Rough KNN LDA Algorithm and Rough KNN QDA Algorithm.
4. These algorithms are quite similar to Fuzzy LDA and Fuzzy QDA algorithms respectively.
5. The only difference is that we use weighted distances for calculating the X and Y values at the end of each algorithm instead of the ordinary distances. 

### Condensed KNN Algorithm

1. This algorithm involves the use of a combination of K-means algorithm along with KNN algorithm.
2. In K-means algorithm, we arrange the data into clusters and remove the outliers in each cluster.
3. Then, we perform the KNN algorithm on the remaining data sets.
4. We can perform two variants of this algorithm – Condensed KNN LDA Algorithm and Condensed KNN QDA Algorithm.
5. These algorithms are quite similar to Condensed KNN.
6. The only difference is that after removing the resulting outliers from K-means process, we perform the KNN LDA and KNN QDA algorithms on the remaining data.

### Constrained KNN Algorithm

1. This algorithm is based on the principle of assigning partial membership in a class to a testing data set.
2. Membership of a testing data set can take on any real values in the interval [0,1]. 
3. We choose k-nearest neighbours that satisfy certain conditions.
4. We can perform two variants of this algorithm – Constrained KNN LDA Algorithm and Constrained KNN QDA Algorithm.
5. Constrained KNN LDA is similar to Condensed KNN LDA algorithm, the only difference being that in this case, we have partial memberships in each class.
6. Constrained KNN QDA is the same as Fuzzy QDA except that at the end we take the percentage of the X and Y values at the end of the algorithm and represent this as the membership of the testing data set in each class

### Accuracy On Diabetes Dataset

Predictions using each of the above were crried out and their accuracy was also computed. The accuracy is computed by comparing the predicted values with the actual values in the partitions taken as our Training Datasets.

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

## Note

I had prepared the below slides for my Final Year Project Review in MSc Mathematics. These slides contain the detailed workings of each of the algorithms as well as their accuracies

<a href = ""> MSc Mathematics Final Year Project Review </a>

Further background behind this project is given in the thesis below which I submitted for my Degree in MSc Mathematics in PDF format.

<a href = "https://github.com/rahulshankariyer/MS_Mathematics_Final_Year_Project/blob/main/RAHUL%20FINAL%20YEAR%20PROJECT%20REPORT.pdf"> HYBRID K-NEAREST NEIGHBOUR AND DISCRIMINANT ANALYSIS FOR PREDICTING MEDICAL DIAGNOSIS IN DECISION SUPPORT SYSTEM </a>
