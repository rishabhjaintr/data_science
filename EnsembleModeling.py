import numpy as np
import pandas as pd
from sklearn import model_selection

class EnsembleModeling:
    '''
    Allows creation of ensemble models through averaging, weighted average,
    stacking, blending, bagging, and boosting.
    '''
    
    def __init__(self, models: list, X: pd.DataFrame or np.ndarray, 
                 y: pd.DataFrame or np.ndarray, 
                 test: pd.DataFrame or np.ndarray):
        self.models = models
        self.X = X
        self.y = y
        self.test = test
    
    def average(self, weights: list or np.ndarray = None):
        '''
        Make predictions using all models passed, average the predictions.
        A weighted average will be computed if the weights are passed.
        The length of weights should equal the number of models.
        It will fit on X, and predict on test data.
        It will return the average prediction, predictions from
        individuals models in the order passed, and also the model objects in
        the format: (average_predictions, all_predictions, fit_models)
        '''
        predictions = np.zeros(shape = 
                               (len(self.test), len(self.models)
                               ))
        fits = [None] * len(self.models)
        for i, model in enumerate(self.models):
            fits[i] = model.fit(self.X, self.y)
            predictions[:, i] = model.predict(self.test)
        
        if weights == None:
            return {'average':np.average(predictions, axis = 1), 
                    'predictions':predictions, 
                    'models': fits}
        else:
            return {'average': np.average(predictions, axis = 1,
                               weights = weights), 
                    'predictions': predictions, 
                    'models': fits}
    
    def stack(self, folds: int):
        '''
        The below process outlines the stacking approach in this method.
        Returns the predictions for test, and modified copies of X, and test.
        
        1. Divide the training set into k folds.
        2. Train first model on k-1 folds, and predict remaining. Repeat for
            other folds.
        3. Train model on entire dataset (not on the predicted column), and
            predict for test set.
        4. Train current model on entire dataset, and predict test set.
        5. Repeat steps 2 to 4 for other models. Sequence will play a role.
        6. The last model's predictions will be used ad final prediction.
        '''
        # extend the X and test copies for model predictions
        X_copy = np.append(self.X, np.zeros(shape = (len(self.X), 
                                                     len(self.models))), 
                  axis = 1)
        test_copy = np.append(self.test, np.zeros(shape = (len(self.test), 
                                                           len(self.models))), 
                  axis = 1)
        
        # index of first model column
        col_num = X_copy.shape[1] - len(self.models) # which is the length of
            # of columns axis of original dataset
        
        # run through each model
        for model_enum, model in enumerate(self.models):
            cv = model_selection.KFold(folds)
            
            # run through each fold
            for (train_index, val_index) in cv.split(X_copy, self.y):
                # fit the model to k-1 fold training data
                model.fit(X_copy[train_index, 0:col_num], self.y[train_index])
                # in the validation rows and the new column, add the prediction
                X_copy[val_index, col_num] = model.predict(X_copy[val_index, 0:col_num])
            
            # now run model on entire dataset (without the predictions just done)
            model.fit(X_copy[:, 0:col_num], self.y)
            # predict test set and add predictions
            test_copy[:, col_num] = model.predict(test_copy[:, 0:col_num])
            # increment col_num so that next model will use this model's predictions
            col_num += 1
        
        # return the modified datasets
        return {'training': X_copy, 
                'testing': test_copy,
		'predictions': test_copy[:, -1]}
    
    def blend(self, validation_fraction: float = 0.15):
        '''
        The process below outlines the blending algorithm. It returns the 
        validation set and modified test set, and predictions for both.
        1. Split training data into train and valid sets.
        2. For each model, fit on train and predict valid and test. Each valid
            and test prediction will be added to the valid and test dataset.
        3. The last model will use the entire validation and test dataset (with
           predicted values) to fit, then predict values for test and valid.
        '''
        # split dataset
        x_train, x_val, y_train, y_val = model_selection.train_test_split(self.X, self.y,
                                                                            test_size = validation_fraction)
        # this is where you can start inputting
        num_cols = np.shape(x_val)[1]
        
        # extend the test set and valid set
        cols =  len(self.models) - 1
        val_copy = np.append(x_val, np.zeros(shape = (len(x_val), cols)), axis = 1)
        test_copy = np.append(self.test, np.zeros(shape = (np.shape(self.test)[0], cols)), axis = 1)
        
        # make predictions on val set (not using the last model)
        for i in range(len(self.models) - 1):
            self.models[i].fit(x_train, y_train)
            val_copy[:, num_cols] = self.models[i].predict(x_val)
            test_copy[:, num_cols] = self.models[i].predict(self.test)
            num_cols += 1
        
        # using validation set with predictions this time, fit last model
        # on validation set and predict
        self.models[-1].fit(val_copy, y_val)
        pred = self.models[-1].predict(test_copy)
        
        return {'predictions': pred,
                'models': self.models}
