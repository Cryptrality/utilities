def initialize(state):
    state.number_offset_trades = 0;


@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):

    bbands = data.bbands(20, 2)
    
    # on erronous data return early (indicators are of NoneType)
    if bbands is None:
        return

    bbands_lower = bbands["bbands_lower"].last
    bbands_upper = bbands["bbands_upper"].last

    current_price = data.close_last

    position_manager = PositionManager(state, data.symbol)
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity    
    position_manager.set_value(float(balance_quoted) * 0.80)
    # print(position_manager.has_position)
    # print(position_manager.position)

    if current_price < bbands_lower and not position_manager.has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", position_manager.position_value, " at current market price: ", data.close_last)
        position_manager.open_market()
        position_manager.double_barrier(0.04, 0.1)

    elif current_price > bbands_upper and position_manager.has_position:
        print("-------")
        logmsg = "Sell Signal: closing {} position with exposure {} at current market price {}"
        print(logmsg.format(data.symbol,float(position_manager.position.exposure),data.close_last))

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


    def __init__(self, state, symbol, include_dust=False):
        position = query_open_position_by_symbol(
            symbol, include_dust=include_dust)
        self.symbol = symbol
        self.position = position
        self.has_position = position is not None
        self.is_pending = False
        try:
            saved_pos = state.positions_manager[self.symbol]
        except TypeError:
            saved_pos = {}
            state.positions_manager = {}
            state.positions_manager[self.symbol] = {}
        except KeyError:
            saved_pos = {}
            state.positions_manager[self.symbol] = {}
        self.position_data = saved_pos
        #TODO self.check_if_waiting()
        #TODO self.check_if_pending()
        if not self.has_position and not self.is_pending:
            self.position_data[self.symbol] = {}
            state.positions_manager[self.symbol] = {}
    
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
        else:
            print("Buy order already placed")
    
    def close_market(self):
        if self.has_position:
            close_position(self.symbol)
            self.cancel_stop_orders()
            self.position_data = {}
            # cleanup state
    
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

    def position_amount(self):
        try:
            amount = float(self.position_data["buy_order"].quantity)
        except Exception:
            amount = None
        return amount
    
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

