# Author = ZephyZephyr#4672
# You can use try except else in your handler (inside your for loop for each symbol):
        try:
            close = data.select_last("close")
        except ValueError:
            # Prevent backtester from failing due to unavailable data
            print("Error: No close data for {} for this interval.".format(symbol))
        else:
            # Do all your handler stuff
