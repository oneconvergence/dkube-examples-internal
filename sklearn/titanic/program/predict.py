import os
import joblib
import numpy as np
import pandas as pd

model_dir = "/model"

def predict():
    test_df = pd.read_csv(os.path.join(model_dir, "test.csv"))
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    predictions = model.predict(test_df)
    output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    print("predictions generated.")


if __name__ == "__main__":
    predict()
