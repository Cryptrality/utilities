'''
This gives the OHLC4 of the last candle. 
'''
@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):
    # get data series as a numpy array
    c = data.select("close")[-1]
    h = data.select("high")[-1]
    l = data.select("low")[-1]
    o = data.select("open")[-1]

    ohlc4 = (o+h+l+c)/4
    print(ohlc4)
'''
To get the whole series convert the data to a numpy array before doing the calculations.
'''