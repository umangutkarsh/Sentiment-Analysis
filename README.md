# Sentiment-Analysis
Introduction: The problem statement of this project is that, there are 1000s of reviews on Amazon of various products. For this project, I've taken the 'Musical Instruments' as the products. The problem with the log reviews is that it becomes very much tedious, to read the reviews and to decide whether to purchase the instrument or not, so in this project is created which basically classifies the reviews on the basis of three sentiments : Negative, Neutral, and Positive using NLP and ML techniques.

The dataset was taken from Kaggle.com, and it contains the reviews of a large number of musical instruments, and the basic idea behind the project is to classify the reviews into three classes
1. Negative
2. Neutral
3. Positive

Various data analysis libraries are used for Data analysis such as numpy, pandas, seaborn etc.

Then some Exploratory Data analysis is done in order to get an idea of the frequency of occurence of the classes of reviews in which they are classified, also since this was an imbalanced dataset, SMOTE technique is used for resampling for improving the accuracies.
In the data preprocessing phase, the the dependent categorical variables are LabelEncoded into numerical values.
Text cleaning and Stemming is done in order to keep only the relevant words for applying NLP techniques. This cleaned text is stored into a list.
The using the CountVectorizer as we only count the number of times a word appears in the document which results in biasing in favour of most frequent words. this ends up in ignoring rare words which could have helped in processing our data more efficiently. Using TfidfVectorizer to consider the overall document weightage of a word. It helps us in dealing with most frequent words. Using it we can penalize them. TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.

Various classification models such as Logistic Regression, Decision Tree, Random Forrest, K Nearest Neighbours, Support Vector Classifier, Naive Bayes' and XGBoost are applied to classify these text, and compute their accuracies.

Initially, 
1. Logistic Regression is giving great results.
2. Decision Tree was over - trained
3. Random Forest gave great results
4. KNN – which works well with imbalanced datasets, was performing the worst. (without resampling, KNN was performing much better with an accuracy of around 87% on the test set)
5. SVC took too much time for training, because of the large dataset and a lot of features.
6. MultinomialNB works well with features which assume vector values, therefore implemented MultinomialNB, other classes of naive bayes' performed poorly
7. XG Boost was one of the best model on this dataset

Issues Faced:
1.	Resampling Issues, model was becoming highly biased (basically overfitting), was not giving proper accuracy on the test set.
2.	KNN – which works well with imbalanced datasets, was performing the worst. (without resampling, KNN was performing much better with an accuracy of around 87% on the test set)
3.	Also, I did resampling after train_test_split, so that the test data doesn’t have duplicated data. 
4.	I used TfidfVectorizer, since because of CountVectorizer, the result was getting highly biased (overfitting).
5.	SVC was taking too much time for training, I could have reduced the training data, but didn’t want to reduce the import features.
6.	To overcome the problem of oversampling, means whether it should be done before or after train_test_split, I used SMOTE which doesn’t have the limitations which RandomOver–sampling has, means that It also prevents duplication, new observations from the minority class will not be identical to original ones. 
7.	I tried the SMOTE after the train_test_split, but it wasn’t giving better results, compared to the when I applied before train_test_split.
8.	KNN was giving, much better accuracy this time.
9.	Finally, I decided to do resampling before the train_test_split, since the performance was much better.
10.	Was trying with GaussianNB, but since MultinomialNB works well with features which assume vector values, therefore implemented MultiNB.
11.	While performing k-fold cross validation, I have set n_jobs = -1 since was taking too much time to compute the accuracies, and same in the case of gridsearchCV.

Then, 
Applying k-fold cross validation for model validation, and to measure in the most relevant way the performance of each model by splitting the training set into a number of train_test folds to obtain the final accuracy as the average of the accuracies obtained on the train_test folds. (In this case, evaluating accuracy by cross-validation)

After cross-validation is done,
Doing some Hyperparameter tuning(using GridSearchCV class) for finding the best version of the models, and to improve the accuracy further.

1. Logistic Regression:
It's a linear classification that supports logistic regression and linear support vector machines. The solver uses a Coordinate Descent (CD) algorithm that solves optimization problems by successively performing approximate minimization along coordinate directions or coordinate hyperplanes.  solver(in Log Reg)
Cfloat, default=1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
Specify the norm of the penalty:
•	'none': no penalty is added;
•	'l2': add a L2 penalty term and it is the default choice;
•	'l1': add a L1 penalty term;
•	'elasticnet': both L1 and L2 penalty terms are added.
Warning
 
Some penalties may not work with some solvers. See the parameter solver below, to know the compatibility between the penalty and solver.

2. Decision Tree cLassifier:
D_tree = criterion{“gini”, “entropy”}, default=”gini” – I tried changing the max_depth parameter, but that led to underfitting.
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
max_depthint, default=None
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

3. Random Forrest Classifier:
R_forrest = n_estimatorsint, default=100  --> playing with max_depth led to underfitting
The number of trees in the forest.
•	Max_depth = max_depth = max number of levels in each decision tree

4. Naive Bayes':
MultinomialNB = alphafloat, default=1.0
Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
Using Laplace smoothing, we can represent P(x'|positive) as, P(x'/positive)= (number of reviews with x' and target_outcome=positive + α) / (N+ α*k) Here, alpha(α) represents the smoothing parameter, K represents the dimensions(no of features) in the data, N represents the number of reviews with target_outcome=positive.

5. KNN:
KNN(slow process) = n_neighbors int, default=5  took a lot of time to tune the hyperparameters, therefore reduced the folds to 3. (tried for larger n also), reducing the n_neighbours speed up the process of grid_search
Number of neighbors to use by default for kneighbors queries.
algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
Algorithm used to compute the nearest neighbors:
•	‘ball_tree’ will use BallTree
•	‘kd_tree’ will use KDTree
•	‘brute’ will use a brute-force search.
•	‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

6. SVC:
[{'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 'kernel':['linear']},
             {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}  tried for both linear and rbf kernel, but took a lot of time, therefore implemented both seperately


Reduced cv to 5 (fir time complexity quadratic, so since large datasets and a lot of features, took too much time)

7. XG Boost:
gamma [default=0, alias: min_split_loss]
a.	Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. (Could have reduced gamma, but taking too much load on the CPU)
b.	range: [0,∞]
booster [default= gbtree ]
c.	Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.

Conclusion:
1. Almost all the accuracies of all the models except for KNN and XGBoost, was improved a bit.
2. KNN could have worked with imbalanced data.
