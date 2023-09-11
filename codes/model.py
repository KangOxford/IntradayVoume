import torch
import numpy as np
array1 = np.concatenate([np.arange(1, 10, 0.01), np.arange(10, 50, 0.1)])
array2 = np.arange(1, 0.001, -0.001)

combined_array = np.array(list(zip(array1, array2))).flatten()


# used for alphas
def regularity_ols(X_train, y_train, X_test, regulator,num):
    if regulator == "CMEM":
        y_pred = X_test['log_x'].to_numpy().flatten()
        # y_pred = X_test['x'].to_numpy().flatten()
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
        import warnings
        from sklearn.exceptions import DataConversionWarning

        # Suppress DataConversionWarning globally
        warnings.filterwarnings("ignore", category=DataConversionWarning)

        # Now you can run your code that generates the warning
        # model.fit(X, y)
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
        # print(y_pred.shape)
        # breakpoint()
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



        # X_train, y_train, X_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy()
        # print(X_train.shape,y_train.shape)
        # X_test_new = np.concatenate([X_train[X_test.shape[0]:,:],X_test])
        # '''#TODO slice the last X_test.shape[0] y_pred'''
        # X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y = normalize_data(X_train, y_train, X_test_new)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # Normalize the data
        # # Convert to PyTorch tensors
        # X_train_tensor, y_train_tensor, X_test_tensor = to_torch_tensors(X_train_scaled, y_train_scaled, X_test_scaled, device)
        # num_stock=1;num_feature=52
        # # num_stock=483;num_feature=52 
        # '''CAUTION remeber to pass right value'''
        # X_train_tensor=X_train_tensor.reshape(num_stock,-1,num_feature).unsqueeze(1)
        # y_train_tensor=y_train_tensor.reshape(num_stock,-1,1)
        # X_test_tensor = X_test_tensor.reshape(num_stock,-1,num_feature).unsqueeze(1)
        # # Initialize the model
        # stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=1200, batch_size=483)
        # # Convert the model's parameters to Double
        # stock_prediction_model.model.double().to(device)
        # # Train and predict
        # stock_prediction_model.train(X_train_tensor, y_train_tensor)
        # y_pred_normalized = stock_prediction_model.predict(X_test_tensor)
        # y_pred = y_pred_normalized[:,-26:,:]
        # y_pred_flatten = y_pred.reshape(-1,1)
        # y_pred_flatten = denormalize_predictions(y_pred_flatten.numpy(), scaler_y)
        # '''caution how y_pred is flattened deserves attention!!!'''
        # return y_pred_flatten
    else:
        raise NotImplementedError


def regularity_nn(X_train, y_train, X_test,y_test, regulator,num):
    assert regulator == "CNN", regulator
    from codes.nn import NNPredictionModel
    from sklearn.preprocessing import MinMaxScaler
    import torch
    import numpy as np

    def normalize_data(X, y):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled,scaler_X, scaler_y

    def to_torch_tensors(X_train, y_train, device):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float64).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float64).to(device)
        return X_train_tensor, y_train_tensor

    def denormalize_predictions(y_pred_normalized, scaler_y):
        y_pred_normalized = y_pred_normalized.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_normalized)
        y_pred = y_pred.reshape(-1)
        return y_pred

    def reshape_tensors(X_train_tensor, y_train_tensor, num_stock, num_feature):
        X_train_tensor = X_train_tensor.reshape(num_stock, -1, num_feature).unsqueeze(1)
        y_train_tensor = y_train_tensor.reshape(num_stock, -1, 1)
        return X_train_tensor, y_train_tensor

    def train_model(X_train_tensor, y_train_tensor, device):
        stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=1200, batch_size=483)
        # stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=12, batch_size=483)
        stock_prediction_model.model.double().to(device)
        stock_prediction_model.train(X_train_tensor, y_train_tensor)
        return stock_prediction_model

    def train_and_predict_with_sliding_window(X_scaled, y_scaled, num_stock, num_feature, device):
        last_preds = []
        for i in range(0, 26):
            # Update the training data to include data up to day i
            X_train_window = X_scaled[i:1300+i, :]
            y_train_window = y_scaled[i:1300+i]
            print(X_train_window.shape, y_train_window.shape)

            # Convert to Torch tensors and Reshape
            X_train_tensor_window, y_train_tensor_window = to_torch_tensors(X_train_window, y_train_window, device)
            X_train_tensor_window, y_train_tensor_window = reshape_tensors(X_train_tensor_window, y_train_tensor_window, num_stock, num_feature)
            
            print(X_train_tensor_window.shape, y_train_tensor_window.shape)
            # Train the model with the new data
            model = train_model(X_train_tensor_window, y_train_tensor_window, device)

            # Prepare the test data for prediction
    
            X_test_window = X_scaled[i+1:1300 + i+1, :]
            y_test_window = y_scaled[i+1:1300 + i+1]
            X_test_tensor_window = torch.tensor(X_test_window, dtype=torch.float64).to(device).reshape(num_stock, -1, num_feature).unsqueeze(1)

            # Make a prediction using the updated model
            y_pred_normalized = model.predict(X_test_tensor_window)
            last_pred = y_pred_normalized[0, -1, 0].item()
            last_preds.append(last_pred)

        return np.array(last_preds).reshape(-1, 1)

    def pred(X_train, y_train, X_test, y_test):
        # Ensure inputs are NumPy arrays
        X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
        
        # Concatenate training and test data for scaling
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        # Normalize data and get scalers
        X_scaled, y_scaled, scaler_X, scaler_y = normalize_data(X, y)
        
        # Device configuration
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Constants for reshaping and sliding window
        NUM_STOCK = 1
        NUM_FEATURE = 52
        
        # Train model and predict with sliding window
        last_preds = train_and_predict_with_sliding_window(X_scaled, y_scaled, NUM_STOCK, NUM_FEATURE, device)
        
        # Denormalize predictions
        last_preds_denorm = denormalize_predictions(last_preds, scaler_y)
        
        return last_preds_denorm
    
    # Call pred function with appropriate data
    last_preds_denorm = pred(X_train, y_train, X_test,y_test)
    return  last_preds_denorm

def model_nn(X_train, y_train, X_test, regulator,num):
    '''take the first 9 days of X_train as X_train_new
    y_train take the last 1 day of y_train as y_train_new
    one day include 26bins(26rows) of data
    the nn file is also in need of modification to forecast 
    from 1,1,1300,52 X 1, 1300,1
    to   1,1,1274,52 X 1,   26,1   '''
    
    assert regulator == "Inception"
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
    X_train_=X_train[:1274,:];y_train_=y_train[-26:,:];X_test_=X_train[26:,:]
    print(X_train_.shape,y_train_.shape,X_test_.shape)
    '''#TODO slice the last X_test.shape[0] y_pred'''
    X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y = normalize_data(X_train_, y_train_, X_test_)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Normalize the data
    # Convert to PyTorch tensors
    X_train_tensor, y_train_tensor, X_test_tensor = to_torch_tensors(X_train_scaled, y_train_scaled, X_test_scaled, device)
    num_stock=1;num_feature=52
    # num_stock=483;num_feature=52 
    '''CAUTION remeber to pass right value'''
    X_train_tensor=X_train_tensor.reshape(num_stock,-1,num_feature).unsqueeze(1)
    y_train_tensor=y_train_tensor.reshape(num_stock,-1,1)
    X_test_tensor = X_test_tensor.reshape(num_stock,-1,num_feature).unsqueeze(1)
    print(X_train_tensor.shape,y_train_tensor.shape,X_test_tensor.shape)
    '''torch.Size([1, 1, 1274, 52]) torch.Size([1, 26, 1]) torch.Size([1, 1, 1274, 52])'''
    # Initialize the model
    stock_prediction_model = NNPredictionModel(learning_rate=0.0002, epochs=1200, batch_size=483)
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