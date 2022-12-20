import pandas as pd
from sklearn.model_selection import train_test_split

class GenerateData:
    
    # Constructor
    def __init__(self, params):
        self.ratio                  = params.get('ratio') or 0.8
        self.test_size              = params.get('test_size') or 0.2
        self.input_file             = params.get('input_file')
        self.output_x_train_file    = params.get('output_x_train_file')
        self.output_y_train_file    = params.get('output_y_train_file')
        self.output_x_test_file     = params.get('output_x_test_file')
        self.output_y_test_file     = params.get('output_y_test_file')

    # Execution
    def run(self):
        df = pd.read_csv(self.input_file)

        self.x = df.iloc[:,:-1]
        self.y = pd.get_dummies(df.iloc[:,-1])  # one-hot encoded y values

        x_train, x_test, y_train, y_test = train_test_split(
            self.x, 
            self.y, 
            test_size=self.test_size, 
            random_state=0
        )

        x_train.to_csv(self.output_x_train_file, index=False)
        x_test.to_csv(self.output_x_test_file, index=False)
        y_train.to_csv(self.output_y_train_file, index=False)
        y_test.to_csv(self.output_y_test_file, index=False)
