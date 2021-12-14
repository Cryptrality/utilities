import numpy as np

def initialize(state):
    pass

@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):
    
    closes = data.select("close")
    signal1 = np.mean(closes[-24:])   # same as a 24 bar SMA

    # plot on the candelstick chart
    with PlotScope.root(data.symbol):
        plot_line("signal1", signal1)

    signal2 = closes[-1] - closes[-6]  
    signal3 = closes[-1] - closes[-24] 

    # plot multiple signals on the same chart
    with PlotScope.group("my_group", data.symbol):
        plot_line("signal2", signal2)
        plot_line("signal3", signal3)

    signal4 = closes[-1] - closes[-12]

    # plot a line on its own
    plot_line("signal4", signal4, data.symbol)