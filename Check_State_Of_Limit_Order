'''
Author: Mark_Trality
'''

def initialize(state):
    state.run_once = False
    state.order=None
    state.order_id = None


@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):

    if not state.run_once:
        state.run_once = True
        limit_price = float(data.close.last * 0.99)
        state.order = order_limit_value(symbol = data.symbol, limit_price = limit_price, value = 100)
        # record the order id
        state.order_id = state.order.id 

    ''' method 1 using order object '''
    if state.order is not None :
        state.order.refresh()
        if state.order.is_filled():
            print(f"Checking order with object. The order {state.order.id} was filled")
            state.order = None
    
    ''' method 2 looking up the order with id '''
    if state.order_id is not None :
        temp_order = query_order(state.order_id)
        if temp_order.is_filled():
            print(f"Check order with id. The order {temp_order.id} was filled")
            state.order_id = None
