import numpy as np


def initialize(state):
    state.number_offset_trades = 0;
    state.next_atr_price = [None, None]


@schedule(interval="15m", symbol="BTCUSDT")
def handler(state, data):

    bbands = data.bbands(20, 2)
    rsi = data.close.rsi(14).ema(10)
    rsi1 = data.close.rsi(6).ema(10)
    ema30 = data.ema(30).last
    atr = data.atr(14).last
    adx_data = data.adx(14)
    adx = adx_data.last
    
    # on erronous data return early (indicators are of NoneType)
    if bbands is None:
        return

    bbands_lower = bbands["bbands_lower"].last
    bbands_upper = bbands["bbands_upper"].last
    bbands_middle = bbands["bbands_middle"].last

    current_price = data.close_last

    bol_b = (current_price - bbands_lower)/(bbands_upper - bbands_lower)

    atr_n_stop_loss = 4
    atr_n_take_profit = 6
    atr_decrease_steps = 1
    stop_loss = atr_to_percent(current_price, atr_n_stop_loss, atr)
    take_profit = atr_to_percent(current_price, atr_n_take_profit, atr)

    rsi_cross_under = cross_under(
        rsi.select("ema")[-2:], rsi1.select("ema")[-2:])
    rsi_cross_over = cross_over(
        rsi.select("ema")[-2:], rsi1.select("ema")[-2:])

    position_manager = PositionManager(state, data.symbol, data.last_time)
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity    
    position_manager.set_value(float(balance_quoted) * 0.80)
    # print(position_manager.has_position)
    # print(position_manager.position)   

    ## Enable this to progressively raise the double barrier
    # if position_manager.has_position and state.next_atr_price[0] < current_price:
    #     if state.next_atr_price[1] > atr_decrease_steps:
    #         new_n = state.next_atr_price[1] - atr_decrease_steps
    #         next_step = current_price + (atr_decrease_steps * atr)
    #         new_stop_loss = atr_to_percent(current_price, atr, n=new_n)
    #         print("advance stop loss to %s atr" % new_n)
    #         position_manager.update_double_barrier(current_price, None, new_stop_loss)
    #         state.next_atr_price[0] = next_step
    #         state.next_atr_price[1] = new_n

    buy_signal = False
    sell_signal = False
    take_last = 90
    last_rsis = rsi1.select("ema")[-take_last:]
    last_adxs = adx_data.select("dx")[-take_last:]

    price_below_lower_band = current_price < bbands_lower
    price_above_upper_band = current_price > bbands_upper

    price_close_lower_band = bol_b < 0.1
    price_close_upper_band = bol_b > 0.8

    ema30_above_mid_band = ema30 > bbands_middle

    last_peaks = detect_peaks(
        last_rsis, mpd=8, edge=None, kpsh=True)
    last_valleys = detect_peaks(
        last_rsis, mpd=8, edge=None, valley=True, kpsh=True)
    rsi_rising = False
    rsi_climbing = False
    rsi_descending = False
    rsi_divergence = False
    rsi_convergence = False

    last_dx_peaks = detect_peaks(
        last_adxs, mpd=8, edge=None, kpsh=True)
    last_dx_valleys = detect_peaks(
        last_adxs, mpd=8, edge=None, valley=True, kpsh=True)
    adx_rising = False
    adx_climbing = False
    adx_descending = False
    adx_divergence = False
    adx_convergence = False

    if len(last_valleys) > 2 and len(last_peaks) > 2:
        rsi_rising =  (
            last_rsis[last_valleys[-1]] > last_rsis[last_valleys[-2]])
        rsi_descending =  (
            last_rsis[last_peaks[-1]] < last_rsis[last_peaks[-2]])
        rsi_climbing = rsi_rising and not rsi_descending
        rsi_convergence = not rsi_rising and rsi_descending
        rsi_divergence = not rsi_rising and not rsi_descending


    if len(last_dx_valleys) > 2 and len(last_dx_peaks) > 2:
        adx_rising =  (
            last_adxs[last_dx_valleys[-1]] > last_adxs[last_dx_valleys[-2]])
        adx_descending =  (
            last_adxs[last_dx_peaks[-1]] < last_adxs[last_dx_peaks[-2]])
        adx_climbing = adx_rising and not adx_descending
        adx_convergence = not adx_rising and adx_descending
        adx_divergence = not adx_rising and not adx_descending


    # if current_price > bbands_middle and position_manager.check_if_waiting():
    #     wating_data, wating_message = position_manager.waiting_data()
    #     if wating_data == "buy":
    #          position_manager.stop_waiting()

    # if (adx_divergence and rsi_divergence) and position_manager.check_if_waiting():
    #     wating_data, wating_message = position_manager.waiting_data()
    #     if wating_data == "buy":
    #          position_manager.stop_waiting()

    if price_close_lower_band and ema30_above_mid_band and not position_manager.has_position:
         position_manager.start_waiting("buy", "waiting RSI confirmation")
    if price_above_upper_band and position_manager.has_position:
         position_manager.stop_waiting()
         position_manager.start_waiting("sell", "waiting RSI confirmation")

    if adx_convergence and position_manager.check_if_waiting():
        wating_data, wating_message = position_manager.waiting_data()
        if wating_data == "buy":
            buy_signal = True
            position_manager.stop_waiting()
    if rsi_descending and position_manager.check_if_waiting():
        wating_data, wating_message = position_manager.waiting_data()
        if wating_data == "sell":
            sell_signal = True
            position_manager.stop_waiting()
    # if rsi_descending and position_manager.check_if_waiting():
    #     wating_data, wating_message = position_manager.waiting_data()
    #     if wating_data == "confirm_rsi_up":
    #         sell_signal = True
    #         position_manager.stop_waiting()   

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

    with PlotScope.group("%B", position_manager.symbol):
        plot("%B", bol_b)

    with PlotScope.group("rsi_cross", position_manager.symbol):
        plot("rsi_long", rsi.last)
        plot("rsi_short", rsi1.last)
        if len(last_peaks) > 0:
            plot("rsi_peaks", last_rsis[last_peaks][-1])
        if len(last_valleys) > 0:
            plot("rsi_valley", last_rsis[last_valleys][-1])

    with PlotScope.group("rsi_divergence", position_manager.symbol):
        plot("rsi_rising", int(rsi_rising))
        plot("rsi_climbing", int(rsi_climbing))
        plot("rsi_descending", int(rsi_descending))
        plot("rsi_divergence", int(rsi_divergence))
        plot("rsi_convergence", int(rsi_convergence))

    with PlotScope.group("adx_peaks", position_manager.symbol):
        plot("adx", adx)
        if len(last_dx_peaks) > 0:
            plot("adx_peaks", last_adxs[last_dx_peaks][-1])
        if len(last_dx_valleys) > 0:
            plot("adx_valley", last_adxs[last_dx_valleys][-1])

    with PlotScope.group("adx_divergence", position_manager.symbol):
        plot("adx_rising", int(adx_rising))
        plot("adx_climbing", int(adx_climbing))
        plot("adx_descending", int(adx_descending))
        plot("adx_divergence", int(adx_divergence))
        plot("adx_convergence", int(adx_convergence))


    if buy_signal and not position_manager.has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", position_manager.position_value, " at current market price: ", data.close_last)
        position_manager.open_market()
        position_manager.double_barrier(take_profit, stop_loss)
        position_manager.start_waiting("confirm_rsi_up", "waiting RSI confirmation")
        next_step = current_price + (atr_decrease_steps * atr)

        state.next_atr_price[0] = next_step
        state.next_atr_price[1] = atr_n_stop_loss

    elif sell_signal and position_manager.has_position:
        print("-------")
        logmsg = "Sell Signal: closing {} position with exposure {} at current market price {}"
        print(logmsg.format(data.symbol,float(position_manager.position_exposure()),data.close_last))

        position_manager.close_market()

    '''
    5) Check strategy profitability
        > print information profitability on every offsetting trade
    '''
    
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
        self.has_position = position is not None
        self.is_pending = False
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

                # reset state and position data
                self.cancel_stop_orders()
                waiting_data = self.position_data["waiting"]
                state.positions_manager[self.symbol]["data"] = self.default_data()
                self.position_data = state.positions_manager[self.symbol]["data"]
                self.position_data["waiting"] = waiting_data
    
    def set_value(self, value):
        try:
            stored_value = self.position_data["value"]
        except KeyError:
            stored_value = None
        if stored_value is None:
           self.position_data["value"] = value 
           self.position_value = value
        else:
            self.position_value = stored_value

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
        if take_profit is None:
            # keep upper as it is
            order_upper_price = float(self.position_data[
                "stop_orders"]["order_upper"].stop_price)
            take_profit = abs(order_upper_price - current_price) / current_price
        if stop_loss is None:
            # Keep low as it is
            order_lower_price = float(self.position_data[
                "stop_orders"]["order_lower"].stop_price)
            stop_loss = abs(order_lower__price - current_price) / current_price
        self.cancel_stop_orders()
        self.double_barrier(take_profit, stop_loss, subtract_fees=subtract_fees)

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
    
    def start_waiting(self, waiting_data=None, waiting_message=None):
        if waiting_data:
            self.position_data["waiting"]["data"] = waiting_data
        if waiting_message:
            self.position_data["waiting"]["message"] = waiting_message
        self.position_data["waiting"]["status"] = True

    def stop_waiting(self, waiting_data=None, waiting_message=None):
        self.position_data["waiting"]["status"] = False

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

def atr_to_percent(close, atr, n=6):
    tp = close + (n * atr)
    return abs(tp - close) / close

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

# Marcos Duarte detect peaks
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] https://github.com/demotu/detecta/blob/master/docs/detect_peaks.ipynb
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    >>> detect_peaks(x, show=True, ax=axs[0], threshold=0.5, title=False)
    >>> detect_peaks(x, show=True, ax=axs[1], threshold=1.5, title=False)
    Version history
    ---------------
    '1.0.7':
        Part of the detecta module - https://pypi.org/project/detecta/  
    '1.0.6':
        Fix issue of when specifying ax object only the first plot was shown
        Add parameter to choose if a title is shown and input a title
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind
