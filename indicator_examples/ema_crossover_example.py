from trality.indicator import ema, crossover, crosses_under, crosses_over
import numpy as np 

def initialize(state):
    state.run = 0

@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):

    ema1 = ema(data.select("close"), 20)
    ema2 = ema(data.select("close"), 50)
    ema_data = np.concatenate((ema1,ema2), axis=0)
    ema_cross = crossover(ema_data)[0,:]
    ema_under = crosses_under(ema_data)[0,:]
    ema_over = crosses_over(ema_data)[0,:]

    with PlotScope.root(data.symbol):
        plot_line("ema1", ema1[0][-1])
        plot_line("ema2", ema2[0][-1])
    plot_line("ema_cross", ema_cross[-1], data.symbol)
    plot_line("ema_over", ema_over[-1], data.symbol)
    plot_line("ema_under", ema_under[-1], data.symbol)