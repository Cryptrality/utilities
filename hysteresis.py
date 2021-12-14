'''
@author:Mark_Trality
@url:https://discord.com/channels/811696600661098507/887753957336838184/908013882822524979

When comparing multiple noisy time series the results can be noisy. Trading often requires making discrete trading decisions based on these noisy signals. 

Using if statements based on noisy inputs can lead to increase transaction costs as positions are open and closed unnecessarily. 

This is an example in code of how to reduce the noise of converting a noisy signal close_price - ema_price to a binary trend signal [-1,1]
By using a ema with a gap using highs and lows the resulting trend signal is much less twitchy since it takes a larger change in the underlying signal before the state changes.

Further reading:
https://en.wikipedia.org/wiki/Hysteresis

'''
def initialize(state):
    state.smooth_trend_signal = 0
    state.raw_trend_signal = 0

@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):

    high_ema = data.high.ema(50)
    low_ema = data.low.ema(50)
    
    # trend signal with hysteresis
    if data.close_last >= high_ema[-1] :
        state.smooth_trend_signal = +1
    elif data.close_last <= low_ema[-1] :
        state.smooth_trend_signal = -1

    close_ema = data.ema(50)
    # standard trend signal
    if data.close_last >= close_ema[-1] :
        state.raw_trend_signal = +1
    elif data.close_last < close_ema[-1] :
        state.raw_trend_signal = -1

    # plot trading position on Symbols chart
    plot_line("raw_signal", state.raw_trend_signal, data.symbol)
    plot_line("hysteresis_signal", state.smooth_trend_signal, data.symbol)