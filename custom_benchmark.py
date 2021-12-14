import numpy as np

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BENCHMARK_WEIGHTS = [0.6, 0.3, 0.1]               # needs to add to one

def initialize(state):
    # trading variables
    state.number_offset_trades = 0

    # variables for tracking benchmark
    state.start_index,state.last_index = 0.0, 0.0
    state.start_portfolio, state.last_portfolio = 0.0, 0.0

    state.benchmark_ret = 0.0
    state.portfolio_ret = 0.0

    if len(SYMBOLS) != len(BENCHMARK_WEIGHTS) :
        raise Exception("Wrong number of benchmark weights")

    # make sure benchmark weights sum to one
    state.benchmark_weights = np.divide(BENCHMARK_WEIGHTS, np.sum(BENCHMARK_WEIGHTS))

@schedule(interval="1h", symbol=SYMBOLS)
def handler(state, dataMap):
    
    # manage trading for each symbol and record prices for benchmark calculation
    symbol_prices = []
    for symbol in SYMBOLS:    # must be processed in order to make sure prices match weights
        data = dataMap.get(symbol)
        if data is not None :
            trading_strategy(state, data)    
            symbol_prices.append(data.close[-1])

    # record benchmark and portfolio value
    portfolio = query_portfolio()    
    if len(symbol_prices) == len(SYMBOLS): # check all prices are avaliable
        index_value = np.dot(symbol_prices, state.benchmark_weights)
        if state.start_index == 0.0:
            state.start_portfolio = portfolio.portfolio_value
            state.start_index = index_value
        state.last_portfolio = portfolio.portfolio_value
        state.last_index = index_value
    
    # record the portfolio and benchmark data
    state.benchmark_ret = (state.last_index - state.start_index) / state.start_index
    state.portfolio_ret = (state.last_portfolio - state.start_portfolio) / state.start_portfolio

    # log the benchmark return
    print(f"benchmark return {100*state.benchmark_ret:.2f}% portfolio return {100*state.portfolio_ret:.2f}%")

    # plot on all the charts
    for symbol, data in dataMap.items():
        with PlotScope.group("returns", symbol):
            plot_line("benchmark return", 100*state.benchmark_ret)
            plot_line("porfolio return", 100*state.portfolio_ret)

def trading_strategy(state, data) :
    
    ema_long = data.ema(40).last
    ema_short = data.ema(20).last
    rsi = data.rsi(14).last
    
    # on erronous data return early (indicators are of NoneType)
    if rsi is None or ema_long is None:
        return

    current_price = data.close_last
    
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity
    buy_value = float(balance_quoted) * 0.80
    
    position = query_open_position_by_symbol(data.symbol,include_dust=False)
    has_position = position is not None

    if ema_short > ema_long and rsi < 40 and not has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", buy_value, " at current market price: ", data.close_last)
        
        order_market_value(symbol=data.symbol, value=buy_value)

    elif ema_short < ema_long and rsi > 60 and has_position:
        print("-------")
        logmsg = "Sell Signal: closing {} position with exposure {} at current market price {}"
        print(logmsg.format(data.symbol,float(position.exposure),data.close_last))

        close_position(data.symbol)
       
    if state.number_offset_trades < portfolio.number_of_offsetting_trades:
        
        pnl = query_portfolio_pnl()
        print("-------")
        print("Accumulated Pnl of Strategy: {}".format(pnl))
        
        offset_trades = portfolio.number_of_offsetting_trades
        number_winners = portfolio.number_of_winning_trades
        print("Number of winning trades {}/{}.".format(number_winners,offset_trades))
        print("Best trade Return : {:.2%}".format(portfolio.best_trade_return))
        print("Worst trade Return : {:.2%}".format(portfolio.worst_trade_return))
        print("Average Profit per Winning Trade : {:.2f}".format(portfolio.average_profit_per_winning_trade))
        print("Average Loss per Losing Trade : {:.2f}".format(portfolio.average_loss_per_losing_trade))
        # reset number offset trades
        state.number_offset_trades = portfolio.number_of_offsetting_trades
