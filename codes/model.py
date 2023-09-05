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
        print(y_pred.shape)
        breakpoint()
        return y_pred
    elif regulator == "XGB":
        import xgboost as xgb
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_lambda=0.1)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_lambda=1.0)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_lambda=10)
        
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_alpha=0.1)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_alpha=1.0)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_alpha=10)
        
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160)
        # model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=160)
        # model = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=160)
        
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=200)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=300)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=400)
        
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.001, n_estimators=160)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=160)
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=1.0, n_estimators=160)
        
        
        # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,reg_lambda=0.1,reg_alpha=0.1)
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160)
        

        print(model.get_params())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()
        return y_pred
    elif regulator == "cnnLstm":
        from codes.nn import NNPredictionModel
        from sklearn.preprocessing import MinMaxScaler
        def normalize_data(X_train, y_train, X_test):
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
            X_test_scaled = scaler_X.transform(X_test)

            return X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y
        def to_torch_tensors(X_train, y_train, X_test, device):
            X_train_tensor = torch.tensor(X_train, dtype=torch.float64).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float64).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float64).to(device)
            
            return X_train_tensor, y_train_tensor, X_test_tensor
        def denormalize_predictions(y_pred_normalized, scaler_y):
            y_pred_normalized = y_pred_normalized.reshape(-1, 1)
            y_pred = scaler_y.inverse_transform(y_pred_normalized)
            y_pred = y_pred.reshape(-1)
            return y_pred
        X_train, y_train, X_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy()
        print(X_train.shape,y_train.shape)
        X_test_new = np.concatenate([X_train[X_test.shape[0]:,:],X_test])
        '''#TODO slice the last X_test.shape[0] y_pred'''
        X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y = normalize_data(X_train, y_train, X_test_new)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Normalize the data
        # Convert to PyTorch tensors
        X_train_tensor, y_train_tensor, X_test_tensor = to_torch_tensors(X_train_scaled, y_train_scaled, X_test_scaled, device)
        num_stock=1;num_feature=52
        # num_stock=483;num_feature=52
        X_train_tensor=X_train_tensor.reshape(num_stock,-1,num_feature).unsqueeze(1)
        y_train_tensor=y_train_tensor.reshape(num_stock,-1,1)
        X_test_tensor = X_test_tensor.reshape(num_stock,-1,num_feature).unsqueeze(1)
        print(X_train_tensor.shape,y_train_tensor.shape,X_test_tensor.shape)
        # Initialize the model
        # stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=1, batch_size=483)
        stock_prediction_model = NNPredictionModel(learning_rate=0.0003, epochs=500, batch_size=483)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=200, batch_size=483)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=200, batch_size=128)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.0005, epochs=100, batch_size=128)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.001, epochs=3, batch_size=128)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.001, epochs=1, batch_size=128)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.001, epochs=3, batch_size=64)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.001, epochs=10, batch_size=64)
        # Convert the model's parameters to Double
        stock_prediction_model.model.double().to(device)
        # Train and predict
        stock_prediction_model.train(X_train_tensor, y_train_tensor)
        y_pred_normalized = stock_prediction_model.predict(X_test_tensor)
        print(y_pred_normalized.shape)
        y_pred = y_pred_normalized[:,-26:,:]
        y_pred_flatten = y_pred.reshape(-1,1)
        print(y_pred_flatten.shape)
        y_pred_flatten = denormalize_predictions(y_pred_flatten.numpy(), scaler_y)
        '''caution how y_pred is flattened deserves attention!!!'''
        # print(y_pred.shape)
        # breakpoint()
        return y_pred_flatten
    else:
        raise NotImplementedError
