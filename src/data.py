import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple

class DataPipeline:
    """Handles data preprocessing, encoding, scaling, and tf.data loading."""
    
    def __init__(self, target_column: str = "y"):
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses tabular data: label encodes categorical features and normalizes numerical ones."""
        data = df.copy()
        
        # Extract target if present
        y = None
        if self.target_column in data.columns:
            y = data[self.target_column]
            data = data.drop(columns=[self.target_column])
        
        # Categorical Encoding
        categorical_cols = [col for col in data.columns if data[col].dtype == "object"]
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.label_encoders[col] = le

        # MinMax Scaling
        scaled_values = self.scaler.fit_transform(data)
        processed_df = pd.DataFrame(scaled_values, columns=data.columns)
        
        return processed_df, y

    @staticmethod
    def create_dataset(df: pd.DataFrame, batch_size: int = 512, buffer_size: int = 100000) -> tf.data.Dataset:
        """Converts DataFrame to batched and shuffled tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(df.values, dtype=tf.float32)
        )
        return dataset.shuffle(buffer_size).batch(batch_size)
