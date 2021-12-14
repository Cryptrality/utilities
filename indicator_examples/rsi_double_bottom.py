__Author__ = "Blackbeard#4057"

RSI_OVERBOUGHT_THRESHOLD = 70
RSI_OVERSOLD_THRESHOLD = 100 - RSI_OVERBOUGHT_THRESHOLD
RSI_MIDDLE_VALUE = 40
RSI_TIMEOUT = 24*5

@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):
# define RSI finite state machine 
    
    start_state = None
    
    while start_state != state.rsi_state:
        start_state = state.rsi_state
    
        is_overbought = rsi[-1] > RSI_OVERBOUGHT_THRESHOLD
        is_oversold_crossed_from_below = (rsi[-1] > RSI_OVERSOLD_THRESHOLD and rsi[-2] < RSI_OVERSOLD_THRESHOLD)
        is_middle_crossed = rsi[-1] > RSI_MIDDLE_VALUE and state.rsi_triggered_time != state.bar_count
        is_timed_out = (state.bar_count - state.rsi_triggered_time) > RSI_TIMEOUT

        if state.rsi_state == 0 and is_oversold_crossed_from_below:
            state.rsi_state = 1
            state.rsi_triggered_time = state.bar_count
        elif state.rsi_state == 1 and is_middle_crossed :
            state.rsi_state = 2
        elif state.rsi_state == 1 and (is_overbought  or is_timed_out): 
            state.rsi_state = 0
        elif state.rsi_state == 2 and (is_overbought or is_timed_out): 
            state.rsi_state = 0
        elif state.rsi_state == 2 and is_oversold_crossed_from_below:
            state.rsi_state = 3
        elif state.rsi_state == 3: 
            state.rsi_state = 0

        # log the state changes
        if start_state != state.rsi_state:
            print(f"state change {start_state} => {state.rsi_state}")
