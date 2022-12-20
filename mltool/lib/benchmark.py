import pandas as pd
from joblib import dump, load
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

class Benchmark:
    
    # Constructor
    def __init__(self, params):
        self.input_model_file   = params.get('input_model_file')
        self.input_test_file    = params.get('input_test_file')
        self.input_labels_file  = params.get('input_labels_file')

    # Execution
    def run(self):
        df_x = pd.read_csv(self.input_test_file)
        df_y = pd.read_csv(self.input_labels_file)

        x = df_x.values
        y = df_y.values

        # Load model
        clf = load(self.input_model_file)

        predictions = clf.predict_proba(x)

        predictions = self.one_hot_encoding(predictions)

        report = classification_report(y, predictions)

        print(report)

    def one_hot_encoding(self, nd_array):
        one_hot_encoding_predictions = nd_array

        for i in range(len(nd_array)):
            max_pred = max(nd_array[i])

            for j in range(len(nd_array[i])):
                one_hot_encoding_predictions[i][j] = 1 if nd_array[i][j] == max_pred else 0

        return one_hot_encoding_predictions

