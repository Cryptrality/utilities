def initialize(state):
    state.trend = 0

@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):
    ema_long = data.ema(50)
    ema_short = data.ema(20)

    # crossover method 1
    trend = +1 if ema_short[-1] > ema_long[-1] else -1   # check the current trend
    cross_over_signal_1 = 0
    if trend != state.trend :
        # check if this is the first bar
        if state.trend != 0 :
            cross_over_signal_1 = trend

            if trend == +1 :
                log("up cross", severity=1)
            elif trend == -1:
                log("dn cross", severity=1)

        state.trend = trend

    # crossover method 2
    # This method can have issues if ema_short are equal ema_long
    cross_over_signal_2 = 0
    if ema_short[-2] < ema_long[-2] and ema_short[-1] > ema_long[-1] :
        cross_over_signal_2 = +1
    elif ema_short[-2] > ema_long[-2] and ema_short[-1] < ema_long[-1] :
        cross_over_signal_2 = -1