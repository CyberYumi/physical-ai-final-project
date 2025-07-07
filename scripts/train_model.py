import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os

def train_model():
    """
    収集したデータ（training_data.csv）を元に、
    模倣学習のAIモデル（ニューラルネットワーク）を訓練し、ファイルに保存する。
    """
    print("Loading training data...")
    file_path = os.path.expanduser('~/my_ros_project/data/training_data.csv')
    df = pd.read_csv(file_path, header=None)
    
    # データの前処理
    df.replace([np.inf, -np.inf], 3.5, inplace=True)
    df.dropna(inplace=True)

    # データを入力（LiDAR）と出力（速度）に分割
    X = df.iloc[:, 2:]
    y = df.iloc[:, :2]

    print("Training the model...")
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=1, activation='relu', solver='adam')
    
    # 学習を実行
    model.fit(X, y)
    print("Model training complete.")
    
    # 学習済みモデルを保存
    model_path = os.path.expanduser('~/my_ros_project/models/imitation_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved to: {model_path}")

if __name__ == '__main__':
    train_model()