import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt


class Prediction:
    # samples and features
    dates = []
    prices = []
    filename = ''

    def __init__(self, filename, dates, prices):
        self.dates = dates
        self.prices = prices
        self.filename = filename

    # get data from csv file of last 30 days
    def get_data(self):
        df = self.clean_data()
        date_series = pd.DatetimeIndex(df['day']).day
        self.dates.extend(date_series)
        self.prices.extend(pd.to_numeric(df['price'].copy()))

        return

    # clean data
    def clean_data(self):
        df = pd.read_csv(self.filename)
        to_drop = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.drop(to_drop, inplace=True, axis=1)
        new_header = ['day', 'price']
        df.columns = new_header

        # taking random sample of data 1/10th
        return df.sample(n=int(len(df.index)/10))

    # prediction by sklearn support vector machine
    def predict_prices(self, x):
        # it need 2D array
        self.dates = pd.DataFrame(self.dates)
        self.dates = np.reshape(self.dates, len(self.dates), 1)

        # create 3 different model to compare
        svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
        svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma=0.1)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')

        # fitting the model
        svr_rbf.fit(self.dates, self.prices)
        svr_lin.fit(self.dates, self.prices)
        svr_poly.fit(self.dates, self.prices)

        # plotting with matplotlib
        plt.scatter(self.dates, self.prices, color='black', label='Data')
        plt.plot(self.dates, svr_rbf.predict(self.dates), color='red', label='RBF Model')
        plt.plot(self.dates, svr_lin.predict(self.dates), color='green', label='Linear Model')
        plt.plot(self.dates, svr_poly.predict(self.dates), color='blue', label='Polynomial Model')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Support Vector Regression - Prediction')
        plt.legend()
        plt.show()

        return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


# call one by another
prediction = Prediction('ONGC.NS.csv', dates=[], prices=[])
prediction.get_data()
# predict price of days
predict_price = prediction.predict_prices(20)
# display the plot
print(predict_price)
