import requests
import pandas
import scipy
import numpy
import sys



TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"

response = []


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    train_data = pandas.read_csv(TRAIN_DATA_URL, header=None).T
    head = train_data.iloc[0]
    train_data = train_data[1:]
    train_data.columns = head
    train_data = train_data.astype('float')
    
    reg = scipy.stats.linregress
    
    slope, intercept, r_value, p_value, std_err = reg(numpy.array(train_data['area']), numpy.array(train_data['price']))
    
    area = numpy.array(area, dtype = 'float')
    predicted_prices = slope*area + intercept
    
    # YOUR IMPLEMENTATION HERE
    return predicted_prices


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
