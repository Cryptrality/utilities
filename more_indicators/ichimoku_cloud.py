'''
Ichimoku Cloud by Mark_Trality

The Ichimoku Cloud is a collection of technical indicators that show support and resistance levels, as well as momentum and trend direction.

read more at: https://www.investopedia.com/terms/i/ichimoku-cloud.asp
'''
def initialize(state):
    state.conversion_base_trend = 0

@schedule(interval="1d", symbol="BTCUSDT")
def handler(state, data):

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    tenkan_sen = (data.high.max(9) + data.low.min(9)) / 2
    tenkan_sen = tenkan_sen[0]

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    kijun_sen = (data.high.max(26) +  data.low.min(26)) / 2
    kijun_sen = kijun_sen[0]

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    idx = min(len(tenkan_sen), len(kijun_sen))  # align two vectors so they can be summed
    senkou_span_A = (tenkan_sen[-idx:] + kijun_sen[-idx:]) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    senkou_span_B = (data.high.max(52) + data.low.min(52)) / 2
    senkou_span_B = senkou_span_B[0]

    # Chikou Span (Lagging Span): Close plotted 26 days in the past
    chikou_span = data.close[:-26]
    chikou_span = chikou_span[0]

    with PlotScope.root(data.symbol):
        plot_line("tenkan_sen", tenkan_sen[-1])
        plot_line("kijun_sen", kijun_sen[-1])
        plot_line("senkou_span_A", senkou_span_A[-1])
        plot_line("senkou_span_B", senkou_span_B[-1])
        plot_line("chikou_span", chikou_span[-1])

    plot_line("ichimoku trend", int(senkou_span_A[-1] > senkou_span_B[-1]), data.symbol)

    # Cross over
    conversion_base_trend = 1 if tenkan_sen[-1] > kijun_sen[-1] else -1
    if conversion_base_trend != state.conversion_base_trend :
        state.conversion_base_trend = conversion_base_trend
        plot_line("cross", conversion_base_trend, data.symbol)

        if conversion_base_trend == 1 :
            log(f"up cross @ {data.close[-1]}", severity=2)
        elif conversion_base_trend == -1 :
            log(f"dn cross @ {data.close[-1]}", severity=2)
    else :
        plot_line("cross", 0, data.symbol)