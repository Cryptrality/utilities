##+------------------------------------------------------------------+
##| Multicoin Template, PositionManager | 15m                                             |
##+------------------------------------------------------------------+


SYMBOLS1 = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
SYMBOLS2 = ["BCHUSDT", "MATICUSDT", "LTCUSDT"]
SYMBOLS3 = ["LUNAUSDT", "SOLUSDT", "RUNEUSDT"]



INTERVAL = "15m"


##+------------------------------------------------------------------+
##| Basic portfolio settings                                         |                                                          |
##+------------------------------------------------------------------+

N_SYMBOLS = 9                   # Specify the number of symbols (it's used
                                # to divide the total amount into N -almost- equal parts)
                                # Max 15 symbols, 5 symbols per handler, you need to
                                # enable extra handlers by removign the comment in the 
                                # decorator line

LEVERAGE = 2                    # Multiplier on the amount when using N_SYMBOLS
                                # eg the balance is 1000 USDT trading with 5 symbols,
                                # with LEVERAGE = 2  the trade will use 400 USDT
                                # instead of 200

FIX_BUY_AMOUNT = 100            # If not None it specify a fix amount for each trade
                                # (overriding other settings)

##+------------------------------------------------------------------+
##| SELL Options                                                     |
##+------------------------------------------------------------------+

ATR_TAKE_PROFIT = 4	            # A multiplier on the ATR value (e.g. 4)
ATR_STOP_LOSS = 6               # A multiplier on the ATR value (e.g. 6)
COLLECT_DATA = False            # if True a python dictionary with the trade data
                                # is printed at the end of each day (You need to specify
                                # additional data to store during the buying process)


import numpy as np
from numpy import greater, less, sum, nan_to_num, exp
from datetime import datetime
from trality.indicator import macd

##+------------------------------------------------------------------+
##| Settings in state (could set different tp/sl for each symbol)    |
##+------------------------------------------------------------------+

def initialize(state):
    state.number_offset_trades = 0
    state.params = {}
    state.balance_quoted = 0
    state.params["DEFAULT"] = {
        "atr_stop_loss": ATR_STOP_LOSS,
        "atr_take_profit": ATR_TAKE_PROFIT,
        "collect_data": COLLECT_DATA}

##+------------------------------------------------------------------+
##| SYMBOL                                                          |
##+------------------------------------------------------------------+

@schedule(interval=INTERVAL, symbol=SYMBOLS1, window_size=200)
def handler(state, data):
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    state.balance_quoted = float(balance_quoted)
    if FIX_BUY_AMOUNT is None:
        buy_value = float(portfolio.portfolio_value) / N_SYMBOLS * LEVERAGE
    else:
        buy_value = FIX_BUY_AMOUNT
    n_pos = 0
    try:
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], buy_value)
    except TypeError:
        handler_main(state, data, buy_value)
    if state.collect_data is not None:
        if int(datetime.fromtimestamp(data[list(data.keys())[0]].last_time / 1000).minute) == 0:
            if int(datetime.fromtimestamp(data[list(data.keys())[0]].last_time / 1000).hour) == 0:
                print(state.collect_data)


##+------------------------------------------------------------------+
##| SYMBOL2 and 3                                                    |
##+------------------------------------------------------------------+

#@schedule(interval=INTERVAL, symbol=SYMBOLS2, window_size=200)
def handler2(state, data):
    portfolio = query_portfolio()
    if FIX_BUY_AMOUNT is None:
        buy_value = float(portfolio.portfolio_value) / N_SYMBOLS * LEVERAGE
    else:
        buy_value = FIX_BUY_AMOUNT
    n_pos = 0
    try:
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], buy_value)
    except TypeError:
        handler_main(state, data, buy_value)

#@schedule(interval=INTERVAL, symbol=SYMBOLS3, window_size=200)
def handler3(state, data):
    portfolio = query_portfolio()
    if FIX_BUY_AMOUNT is None:
        buy_value = float(portfolio.portfolio_value) / N_SYMBOLS * LEVERAGE
    else:
        buy_value = FIX_BUY_AMOUNT
    n_pos = 0
    try:
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], buy_value)
    except TypeError:
        handler_main(state, data, buy_value)
    if state.number_offset_trades < portfolio.number_of_offsetting_trades:
        
        pnl = query_portfolio_pnl()
        offset_trades = portfolio.number_of_offsetting_trades
        number_winners = portfolio.number_of_winning_trades
        msg_data = {"pnl": str(pnl), "number_winners": number_winners,
            "offset_trades": offset_trades, "best_trade_return": portfolio.best_trade_return * 100,
            "worst_trade_return": portfolio.worst_trade_return * 100,
            "average_profit_per_winning_trade": portfolio.average_profit_per_winning_trade,
            "average_loss_per_losing_trade": portfolio.average_loss_per_losing_trade}
        msg = (
            "-------\n"
            "Accumulated Pnl of Strategy:\n   %(pnl)s\n"
            "Number of winning trades %(number_winners)i / %(offset_trades)i.\n"
            "Best trade Return : %(best_trade_return).2f%%\n"
            "Worst trade Return : %(worst_trade_return).2f%%\n"
            "Average Profit per Winning Trade : %(average_profit_per_winning_trade).2f\n"
            "Average Loss per Losing Trade : %(average_loss_per_losing_trade).2f\n")
        print(msg % msg_data)
        # reset number offset trades
        state.number_offset_trades = portfolio.number_of_offsetting_trades



####
def handler_main(state, data, amount):

    if data is None:
        return

    symbol = data.symbol

    #--------------------------------------------#
    # Get Parameters and init variables in state #
    #--------------------------------------------#

    params = get_default_params(state, symbol)
    atr_stop_loss = params["atr_stop_loss"]
    atr_take_profit = params["atr_take_profit"]
    collect_data = params["collect_data"]

    #------------#
    # Indicators #
    #------------#

    rsi_data = data.rsi(14)
    rsi = rsi_data.last
    adx_data = data.adx(14)
    adx = adx_data.last
    atr_data = data.atr(12)
    atr = atr_data.last
    engulfing = data.cdlengulfing().last

    ema_long_data = data.ema(40)
    ema_short_data = data.ema(10)

    ema_long = ema_long_data.last
    ema_short = ema_short_data.last

    current_price = float(data.close_last)
    current_open = float(data.open_last)
    last_price = float(data.select("close")[-2])

    bbands = data.bbands(
        20, 2)
    kbands = keltner_channels(
        data, 20, 20, 2, 40)



    bbands_above_keltner_up = bbands.select(
        'bbands_upper')[-1] > kbands['high'][-1]
    bbands_below_keltner_low = bbands.select(
        'bbands_lower')[-1] < kbands['low'][-1]



    #--------------------------#
    # Init the PositionManager #
    #--------------------------#

    position_manager = PositionManager(state, data.symbol, data.last_time)
    balance_quoted = state.balance_quoted
    position_manager.set_value(float(amount), update=True)

    #-------------------------------------#
    # Assess Stop loss/take profit limits #
    #-------------------------------------#


    if atr_stop_loss is not None:
        stop_loss, sl_price = atr_tp_sl_percent(
            float(current_price), float(atr), atr_stop_loss, False)
    if atr_take_profit is not None:
        take_profit, tp_price = atr_tp_sl_percent(
            float(current_price), float(atr), atr_take_profit, True)


    """Place stop loss for manually added positions"""
    if position_manager.has_position and not position_manager.is_stop_placed():
        position_manager.double_barrier(take_profit, stop_loss)


    """
    Check entry and stop loss values
    """

    try:
        sl_price = position_manager.position_data[
            "stop_orders"]["order_lower"].stop_price
    except Exception:
         sl_price = None


    """
    Basic Strategy
    """

    buy_signal = False
    sell_signal = False

    ### IMPORTANT ###
    ### This is not a strategy! just a random line
    ### to create believable sell/buy signals to test the bot

    if not position_manager.has_position and (
        current_open < current_price and current_open > kbands[
            "low"][-1] and last_price < kbands[
                "low"][-2] and bbands_below_keltner_low):
        buy_signal = True

    if position_manager.has_position and current_price > kbands["high"][-1] and bbands_above_keltner_up:
        sell_signal = True


    #--------------------------------------------#
    # Feedback on PnL and data collection prints #
    #--------------------------------------------#

    if position_manager.pnl_changed:
        summary_performance = state.positions_manager[symbol]["summary"]
        perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
        print(
            perf_message % (
                symbol, summary_performance['winning'],
                summary_performance['tot'],
                float(summary_performance['pnl'])))
        perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
    if int(datetime.fromtimestamp(data.last_time / 1000).minute) == 0:
        if int(datetime.fromtimestamp(data.last_time / 1000).hour) == 0:
            summary_performance = state.positions_manager[symbol]["summary"]
            perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
            print(
                perf_message % (
                    symbol, summary_performance['winning'],
                    summary_performance['tot'],
                    float(summary_performance['pnl'])))
            perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")


    #-------------------------------------------------#
    # Assess available balance and target trade value #
    #-------------------------------------------------#

    skip_buy = False
    if balance_quoted <= position_manager.position_value and not position_manager.has_position:
        if balance_quoted < 20:
            print(
                "WARNING: Balance quoted (%s) is less than "
                "the minimum buy amount (%s)." % (
                    balance_quoted, position_manager.position_value))
            skip_buy = True
        else:
            position_manager.set_value(
                balance_quoted * 0.95, update=True)

    #--------------#
    # Plot section #
    #--------------#

    with PlotScope.root(symbol):
        plot("k_ema", kbands["middle"][-1])
        plot("k_upper", kbands["high"][-1])
        plot("k_lower", kbands["low"][-1])


    try:
        tp_price = position_manager.position_data[
            "stop_orders"]["order_upper"].stop_price
        sl_price = position_manager.position_data[
            "stop_orders"]["order_lower"].stop_price
        with PlotScope.root(position_manager.symbol):
            plot("tp", tp_price)
            plot("sl", sl_price)
    except Exception:
        pass


    with PlotScope.group("pnl", symbol):
        plot("pnl", float(state.positions_manager[
            symbol]["summary"]['pnl']))


    #----------------------#
    # Buy/Sell instruction #
    #----------------------#
    """
    Place all the indicator and info you want to save for the trade in the 
    indicators_data object (use built in type, numpy data will not be printed)
    properly to export by copy/paste
    """

    indicators_data = {}

    if buy_signal and not position_manager.has_position:
        signal_msg_data = {
            "symbol": symbol,
            "value": position_manager.position_value,
            "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Buy Signal: creating market order for %(symbol)s\n"
            "Buy value: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        skip_msg = (
            "++++++\n"
            "Skip buy market order for %(symbol)s\n"
            "Not enough balance: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        if skip_buy is False:
            state.balance_quoted -= position_manager.position_value
            position_manager.open_market()
            position_manager.double_barrier(take_profit, stop_loss)
            if collect_data:
                position_manager.collect_data(state, indicators_data)
            print(signal_msg % signal_msg_data)
        else:
            print(skip_msg % signal_msg_data)


    elif sell_signal and position_manager.has_position:
        signal_msg_data = {
            "symbol": symbol,
            "amount": position_manager.position_exposure(),
            "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Sell Signal: creating market order for %(symbol)s\n"
            "Sell amount: %(amount)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        position_manager.close_market()


##+------------------------------------------------------------------+
##| methods and helpers                                              |
##+------------------------------------------------------------------+

def get_default_params(state, symbol):
    default_params = state.params["DEFAULT"]
    try:
        params = state.params[symbol]
        for key in default_params:
            if key not in params.keys():
                params[key] = default_params[key]
    except KeyError:
        params = default_params
    return params


def cross_over(x, y):
    if y[1] < x[1]:
        return False
    else:
        if x[0] > y[0]:
            return True
        else:
            return False


def cross_under(x, y):
    if y[1] > x[1]:
        return False
    else:
        if x[0] < y[0]:
            return True
        else:
            return False


def price_to_percent(close, price):
    return abs(price - close) / close

def atr_tp_sl_percent(close, atr, n=6, tp=True):
    if tp is True:
        tp = close + (n * atr)
    else:
        tp = close - (n * atr)
    return (price_to_percent(close, tp), tp)


class PositionManager:
    """
    A simple helper to manage positions boilerplate functionalities.
    It wraps around and extends the Trality position functionalities.
        - Query for open position (query_open_position_by_symbol)
        - Store the orders relative to the current position in state
          - A position can be declared without the orders to be filled/placed yet:
            waiting confirmation (TODO) or waiting filling limit orders (TODO)
          - Stop orders can be declared by the same class that opens the position
            getting all the info and storing the stop orders objects with the
            current corresponding symbol-position with the other orders
        - Keep track of per-symbols pnl and winning/losing records (extending the
          base position implementation where it's impossible to record pnl of a position
          terminated by a stop order before the first candle)
    propose syntax:
        position_manager = PositionManager(state, "BTCUSDT")
    query if has position:
        position_manager.has_position
    Set a value to the position
        position_manager.set_value(position_value)
    open the position:
        position_manager.open_market()
    save things in state without placing orders (eg waiting for confirmation):
        position_manager.open_wait_confirmation()
        # this will set True to the attribute position_manager.is_pending 
    open a position with a limit order and deal with the pending status:
        position_manager.open_limit(price_limit)
        .... need to think a clean pattern for limit/if_touched/trailing orders
    check if stop orders are present and add them if not:
        if not position_manager.has_double_barrier:
            position_manager.double_barrier(
                stop_loss=0.1, take_profit=0.05)
    close the position:
        position_manager.close_market()
    """


    def __init__(self, state, symbol, timestamp, include_dust=False):
        position = query_open_position_by_symbol(
            symbol, include_dust=include_dust)
        self.symbol = symbol
        self.timestamp = int(timestamp)
        self.has_position = position is not None
        self.is_pending = False
        self.pnl_changed = False
        try:
            self.position_data = state.positions_manager[self.symbol]["data"]
        except TypeError:
            state.positions_manager = {}
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "last_closure_type": None, "last_pnl": 0, "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        except KeyError:
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "last_closure_type": None, "last_pnl": 0, "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        if self.has_position:
            self.position_data["position"] = position
            if self.position_data["buy_order"] is None:
                # Potential manual buy or existing positions
                # when the bot was started
                order_id = self.position_data["position"].order_ids[-1]
                self.position_data["buy_order"] = query_order(order_id)

        #TODO self.check_if_waiting()
        #TODO self.check_if_pending()
        if not self.has_position and not self.is_pending:
            if self.position_data["buy_order"] is not None:
                stop_orders_filled = self.is_stop_filled()
                if stop_orders_filled:
                    state.positions_manager[
                        self.symbol]["summary"][
                            "last_closure_type"] = stop_orders_filled["side"]
                else:
                    state.positions_manager[
                        self.symbol]["summary"][
                            "last_closure_type"] = "rule"                    
                try:
                    closed_position = self.position_data["position"]
                except KeyError:
                    closed_position = None
                if closed_position is not None:
                    closed_position = query_position_by_id(closed_position.id)
                    pnl = float(closed_position.realized_pnl)
                    if pnl > 0:
                        state.positions_manager[self.symbol]["summary"]["winning"] += 1
                    state.positions_manager[self.symbol]["summary"]["tot"] += 1
                    state.positions_manager[self.symbol]["summary"]["pnl"] += pnl
                    state.positions_manager[self.symbol]["summary"]["last_pnl"] = pnl
                    try:
                        if state.collect_data:
                            state.collect_data[
                                self.symbol][
                                    str(closed_position.entry_time)]["pnl"] = pnl
                    except KeyError:
                        pass
                    self.pnl_changed = True
                else:
                    if stop_orders_filled:
                        sold_value = float((
                            stop_orders_filled[
                                "order"].executed_quantity * stop_orders_filled[
                                    "order"].executed_price) - stop_orders_filled[
                                        "order"].fees)
                        pnl = sold_value - self.position_value()
                        if pnl > 0:
                            state.positions_manager[self.symbol]["summary"]["winning"] += 1
                        state.positions_manager[self.symbol]["summary"]["tot"] += 1
                        state.positions_manager[self.symbol]["summary"]["pnl"] += pnl
                        state.positions_manager[self.symbol]["summary"]["last_pnl"] = pnl
                        try:
                            if state.collect_data:
                                state.collect_data[
                                    self.symbol][str(
                                        stop_orders_filled["order"].created_time)]["pnl"] = pnl
                        except KeyError:
                            pass
                        self.pnl_changed = True

                # reset state and position data
                self.cancel_stop_orders()
                waiting_data = self.position_data["waiting"]
                state.positions_manager[self.symbol]["data"] = self.default_data()
                self.position_data = state.positions_manager[self.symbol]["data"]
                self.position_data["waiting"] = waiting_data
    
    def set_value(self, value, update=False):
        try:
            stored_value = self.position_data["value"]
        except KeyError:
            stored_value = None
        if stored_value is None:
           self.position_data["value"] = value 
           self.position_value = value
        else:
            self.position_value = stored_value
        if update:
            self.position_value = value

    def get_entry_price(self):
        entry_price = None
        if self.has_position:
            try:
                entry_price = float(
                    self.position_data["position"].entry_price)
            except Exception:
                pass
        return entry_price

    def open_market(self):
        try:
            buy_order = self.position_data["buy_order"]
        except KeyError:
            buy_order = None
        if buy_order is None:        
            buy_order = order_market_value(
                symbol=self.symbol, value=self.position_value)
            self.position_data["buy_order"] = buy_order
            #self.__update_state__()
        else:
            print("Buy order already placed")
        # if self.check_if_waiting():
        #     self.stop_waiting()

    def close_market(self):
        if self.has_position:
            close_position(self.symbol)
            self.cancel_stop_orders()
        # if self.check_if_waiting():
        #     self.stop_waiting()

    def double_barrier(self, take_profit, stop_loss, subtract_fees=False):
        try:
            stop_orders = self.position_data["stop_orders"]
        except KeyError:
            stop_orders = {
                "order_upper": None, "order_lower": None}
        if stop_orders["order_upper"] is None:
            amount = self.position_amount()
            if amount is None:
                print("No amount to sell in position")
                return
            with OrderScope.one_cancels_others():
                stop_orders["order_upper"] = order_take_profit(
                    self.symbol, amount, take_profit, subtract_fees=subtract_fees)
                stop_orders["order_lower"] = order_stop_loss(
                    self.symbol, amount, stop_loss, subtract_fees=subtract_fees)
            if stop_orders["order_upper"].status != OrderStatus.Pending:
                errmsg = "make_double barrier failed with: {}"
                raise ValueError(errmsg.format(stop_orders["order_upper"].error))
            self.position_data["stop_orders"] = stop_orders
        else:
            print("Stop orders already exist")

    def is_stop_filled(self):
        try:
            stop_orders = self.position_data["stop_orders"]
            stop_loss = stop_orders["order_lower"]
            take_profit = stop_orders["order_upper"]
        except KeyError:
            return None
        if stop_loss is not None:
            stop_loss.refresh()
            if stop_loss.is_filled():
                return {"side": "stop_loss", "order": stop_loss}
        if take_profit is not None:
            take_profit.refresh()
            if take_profit.is_filled():
                return {"side": "take_profit", "order": take_profit}

    def is_stop_placed(self):
        try:
            stop_orders = self.position_data["stop_orders"]
            stop_loss = stop_orders["order_lower"]
            take_profit = stop_orders["order_upper"]
        except KeyError:
            return False
        if stop_loss is None and take_profit is None:
            return False
        else:
            return True

    def update_double_barrier(self, current_price, take_profit=None, stop_loss=None, subtract_fees=False):
        success = True
        if take_profit is None:
            # keep upper as it is
            try:
                order_upper_price = float(self.position_data[
                    "stop_orders"]["order_upper"].stop_price)
                take_profit = abs(
                    order_upper_price - current_price) / current_price
            except:
                success = False
        if stop_loss is None:
            # Keep low as it is
            try:
                order_lower_price = float(self.position_data[
                    "stop_orders"]["order_lower"].stop_price)
                stop_loss = abs(
                    order_lower__price - current_price) / current_price
            except:
                success = False
        if success:
            self.cancel_stop_orders()
            self.double_barrier(
                take_profit, stop_loss, subtract_fees=subtract_fees)
        else:
            print("update stop limits failed")

    def position_amount(self):
        try:
            amount = float(self.position_data["buy_order"].quantity)
        except Exception:
            amount = None
        return amount

    def position_value(self):
        try:
            buy_order = self.position_data["buy_order"]
            buy_order.refresh()
            value = float(
                (buy_order.executed_quantity * buy_order.executed_price) - buy_order.fees)
        except KeyError:
            value = None
        return value

    def position_exposure(self):
        try:
            exposure = float(self.position_data["position"].exposure)
        except KeyError:
            exposure = None
        return exposure       
    
    def cancel_stop_orders(self):
        try:
            stop_orders = self.position_data["stop_orders"]
        except KeyError:
            stop_orders = {
                "order_upper": None, "order_lower": None}
        for stop_level in stop_orders:
            if stop_orders[stop_level] is not None:
                try:
                    cancel_order(stop_orders[stop_level].id)
                    stop_orders[stop_level] = None
                except Exception:
                    pass
        self.position_data["stop_orders"] = stop_orders
    
    def collect_data(self, state, data_dict):
        if state.collect_data is None:
            state.collect_data = {}
        try:
            state.collect_data[self.symbol]["%s" % self.timestamp] = data_dict
        except KeyError:
            state.collect_data[self.symbol] = {}
            state.collect_data[self.symbol]["%s" % self.timestamp] = data_dict        
    
    def start_waiting(self, waiting_data=None, waiting_message=None):
        if waiting_data:
            self.position_data["waiting"]["data"] = waiting_data
        if waiting_message:
            self.position_data["waiting"]["message"] = waiting_message
        self.position_data["waiting"]["status"] = True

    def stop_waiting(self, waiting_data=None, waiting_message=None):
        self.position_data["waiting"]["status"] = False
        self.position_data["waiting"]["data"] = waiting_data
        self.position_data["waiting"]["message"] = waiting_message

    def check_if_waiting(self):
        if self.position_data["waiting"]["status"] is None:
            return False
        else:
            return self.position_data["waiting"]["status"]

    def waiting_data(self):
        return (
            self.position_data["waiting"]["data"],
            self.position_data["waiting"]["message"])

    def default_data(self):
        return {
            "stop_orders": {"order_upper": None, "order_lower": None},
            "position": None,
            "waiting": {"status": None, "data": None, "message": None},
            "buy_order": None,
            "value": None
        }



def keltner_channels(data, period=20, atr_period=10, kc_mult=2, take_last=50):
    """
    calculate keltner channels mid, up and low values
    """
    ema = data.close.ema(period).select('ema')
    atr = data.atr(atr_period).select('atr')
    high = ema[-take_last:] + (kc_mult * atr[-take_last:])
    low = ema[-take_last:] - (kc_mult * atr[-take_last:])
    return {'middle': ema, 'high': high, 'low': low}



def get_yesterday_daily_candle(data):
    today = datetime.fromtimestamp(data.times[-1] / 1000).day
    yesterday = None
    op = None
    cl = None
    hi = None
    lo = None
    for i in range(len(data.times) - 1, -1, -1):
        day = datetime.fromtimestamp((data.times[i] / 1000)).day
        if day != today and yesterday is None:
            yesterday = datetime.fromtimestamp((data.times[i] / 1000)).day
            cl = float(data.close[i])
            hi = float(data.high[i])
            lo = float(data.low[i])
        elif yesterday is not None and day == yesterday:
            op = float(data.open[i])
            if float(data.low[i]) < lo:
                lo = float(data.low[i])
            if float(data.high[i]) > hi:
                hi = float(data.high[i])
        elif yesterday is not None and day != yesterday and day != today:
            #print({"today": today, "yesterday": yesterday, "last_data": day})
            return({"high": hi, "low": lo, "open": op, "close": cl})


def hourly_candle(data):
    op = None
    cl = None
    hi = None
    lo = None
    vo = None
    for i in range(len(data.times)):
        minute = datetime.fromtimestamp((data.times[i] / 1000)).minute
        if minute == 15:
            op = float(data.open[i])
            hi = float(data.high[i])
            lo = float(data.low[i])
            vo = float(data.volume[i])
        elif vo and minute == 0:
            cl = float(data.close[i])
            if float(data.low[i]) < lo:
                lo = float(data.low[i])
            if float(data.high[i]) > hi:
                hi = float(data.high[i])
            vo += float(data.volume[i])
            yield({"high": hi, "low": lo, "open": op, "close": cl, "volume": vo})
            op = None
            cl = None
            hi = None
            lo = None
            vo = None
        elif vo:
            if float(data.low[i]) < lo:
                lo = float(data.low[i])
            if float(data.high[i]) > hi:
                hi = float(data.high[i])
            vo += float(data.volume[i])