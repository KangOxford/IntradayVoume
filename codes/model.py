import torch
import numpy as np
array1 = np.concatenate([np.arange(1, 10, 0.01), np.arange(10, 50, 0.1)])
array2 = np.arange(1, 0.001, -0.001)

combined_array = np.array(list(zip(array1, array2))).flatten()


# used for alphas
def regularity_ols(X_train, y_train, X_test, regulator,num):
    if regulator == "None":
        y_pred = X_test.to_numpy().flatten()
        return y_pred
    elif regulator == "OLS":
        # print("OLS")
        import statsmodels.api as sm
        def ols_with_summary(X, y):
            X = sm.add_constant(X, has_constant='add')
            results = sm.OLS(y, X).fit()
            return results

        model = ols_with_summary(X_train, y_train)
        X = sm.add_constant(X_test, has_constant='add')
        y_pred = model.predict(X).values
        # assert type(y_pred) == np.float64
        return y_pred
    elif regulator in ["Lasso", "Ridge"]:
        # print("LASSO / RIDGE")
        def find_best_regularity_alpha(X_train, y_train):
            if regulator == "Lasso":
                from sklearn.linear_model import LassoCV
                model = LassoCV(random_state=0, max_iter=10000000)
            if regulator == "Ridge":
                from sklearn.linear_model import RidgeCV
                model = RidgeCV(alphas=combined_array)
            model.fit(X_train, y_train)
            return model.alpha_

        best_regularity_alpha = find_best_regularity_alpha(X_train, y_train)
        # print(best_regularity_alpha) #$
        if regulator == "Lasso":
            from sklearn.linear_model import Lasso
            reg = Lasso(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        if regulator == "Ridge":
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        reg.fit(X_train, y_train)
        # X = pd.DataFrame(X_test).T
        # y_pred = reg.predict(X)
        y_pred = reg.predict(X_test)
        y_pred = y_pred.flatten()
        return y_pred
    elif regulator == "XGB":
        import xgboost as xgb
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()
        return y_pred
    elif regulator == "cnnLstm":
        from codes.nn import NNPredictionModel
        # Convert Pandas DataFrame to PyTorch tensor (Double)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float64).to(device)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float64).to(device)
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float64).to(device)

        # Initialize the model
        stock_prediction_model = NNPredictionModel(numFeature=X_train.shape[1], numStock=num)

        # Convert the model's parameters to Double
        stock_prediction_model.model.double().to(device)

        # Train and predict
        stock_prediction_model.train(X_train, y_train)
        y_pred = stock_prediction_model.predict(X_test)  # y_pred as the output
        y_pred = y_pred.cpu().numpy()
        return y_pred
    else:
        raise NotImplementedError
