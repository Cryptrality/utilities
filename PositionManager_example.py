def initialize(state):
    state.number_offset_trades = 0;
    state.next_atr_price = [None, None]


@schedule(interval="15m", symbol="BTCUSDT")
def handler(state, data):

    bbands = data.bbands(20, 2)
    atr = data.atr(14).last
    
    # on erronous data return early (indicators are of NoneType)
    if bbands is None:
        return

    bbands_lower = bbands["bbands_lower"].last
    bbands_upper = bbands["bbands_upper"].last

    current_price = data.close_last
    stop_loss = atr_to_percent(current_price, 6, atr)
    take_profit = atr_to_percent(current_price, 6, atr)

    position_manager = PositionManager(state, data.symbol, data.last_time)
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity    
    position_manager.set_value(float(balance_quoted) * 0.80)
    # print(position_manager.has_position)
    # print(position_manager.position)
    if position_manager.has_position and state.next_atr_price[0] < current_price:
        if state.next_atr_price[1] > 1:
            new_n = state.next_atr_price[1] - 1
            next_step = current_price + (1 * atr)
            new_stop_loss = atr_to_percent(current_price, atr, n=new_n)
            print("advance stop loss to %i atr" % new_n)
            position_manager.update_double_barrier(current_price, None, new_stop_loss)
            state.next_atr_price[0] = next_step
            state.next_atr_price[1] = new_n

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

    if current_price < bbands_lower and not position_manager.has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", position_manager.position_value, " at current market price: ", data.close_last)
        position_manager.open_market()
        position_manager.double_barrier(take_profit, stop_loss)
        next_step = current_price + (1 * atr)

        state.next_atr_price[0] = next_step
        state.next_atr_price[1] = 6

    elif current_price > bbands_upper and position_manager.has_position:
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
                    "positions": [], "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        except KeyError:
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "positions": [], "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        if self.has_position:
            self.position_data["position"] = position
        #TODO self.check_if_waiting()
        #TODO self.check_if_pending()
        if not self.has_position and not self.is_pending:
            if self.position_data["buy_order"] is not None:
                try:
                    closed_position = self.position_data["position"]
                except KeyError:
                    closed_position = None
                if closed_position is not None:
                    closed_position = query_position_by_id(closed_position.id)
                    pnl = float(closed_position.realized_pnl)
                    if pnl > 0:
                        state.positions_manager[self.symbol]["summary"]['winning'] += 1
                    state.positions_manager[self.symbol]["summary"]['tot'] += 1
                    state.positions_manager[self.symbol]["summary"]['pnl'] += pnl
                else:
                    stop_orders_filled = self.is_stop_filled()
                    if stop_orders_filled:
                        sold_value = float((
                            stop_orders_filled[
                                "order"].executed_quantity * stop_orders_filled[
                                    "order"].executed_price) - stop_orders_filled[
                                        "order"].fees)
                        pnl = sold_value - self.position_value()
                        if pnl > 0:
                            state.positions_manager[self.symbol]["summary"]['winning'] += 1
                        state.positions_manager[self.symbol]["summary"]['tot'] += 1
                        state.positions_manager[self.symbol]["summary"]['pnl'] += pnl
                # reset state and position data
                state.positions_manager[self.symbol]["data"] = self.default_data()
                self.position_data = state.positions_manager[self.symbol]["data"]
    
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
    
    def close_market(self):
        if self.has_position:
            close_position(self.symbol)
            self.cancel_stop_orders()

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
    
    def default_data(self):
        return {
            "stop_orders": {"order_upper": None, "order_lower": None},
            "position": None,
            "buy_order": None,
            "value": None
        }

def atr_to_percent(close, atr, n=6):
    tp = close + (n * atr)
    return abs(tp - close) / close
