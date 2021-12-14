@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):
    # get data series as a numpy array
    c = data.select("close")
    h = data.select("high")
    l = data.select("low")
    o = data.select("open")

    ohlc4 = (o+h+l+c)/4
    print(ohlc4)