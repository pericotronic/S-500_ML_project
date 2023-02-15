import pickle
import pandas as pd
import numpy as np
import gzip

model_file = r"Alumno\3-Machine_Learning\Entregas\ML_project\src\model\model_230215232253.pickle.gz"

# Unzip the model file
with gzip.open(model_file, 'rb') as f_in:
    with open(r"Alumno\3-Machine_Learning\Entregas\ML_project\src\model\model_230215232253.pickle", 'wb') as f_out:
        f_out.write(f_in.read())

# Load the model from the file
with open(r"Alumno\3-Machine_Learning\Entregas\ML_project\src\model\model_230215232253.pickle", 'rb') as f:
    model = pickle.load(f)

# Load the test data
test_data = pd.read_csv(r"Alumno\3-Machine_Learning\Entregas\ML_project\src\4. data\test2.csv")

# Make predictions using the model
predictions = model.predict(test_data)

# Save the predictions as a CSV file
predictions_df = pd.DataFrame({"target": predictions})
predictions_df.to_csv("Alumno/3-Machine_Learning/Entregas/ML_project/src/predictions.csv", index=False)
