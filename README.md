# random-forest-knn
### Applying a random forest KNN clustering algorithm to predict binary classification on a real-world data set.

This model makes predictions using a random forest classifier from Sci-kit Learn's RandomForestClassifier library. The project uses actualy world data, with missing values imputed in the flat files. The flat files themselves are NOT provdided as the model is being trained for real data prediction.

### How it works
  forest_knn makes use of Sklearn's provided train_test_split to split the 9000 row training data into 10% 90% test-train split, respectively, and then makes a prediction and scores accuracy on train/test set.

 -forest_knn_GridSearchCV.py returns the best optimized parameters for the random forest model predictor from a set of given parameters using Grid Search Cross-Validation.

