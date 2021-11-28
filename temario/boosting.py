import pandas as pd 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datasets/heart.csv')

    x = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    boost = GradientBoostingClassifier(loss='exponential',learning_rate=0.15, n_estimators=100, max_depth=5).fit(x_train, y_train)
    boost_pred = boost.predict(x_test)
    print("="*64)
    print(accuracy_score(boost_pred, y_test))