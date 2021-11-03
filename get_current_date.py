import datetime
import pytz

def initialize(state):
    state.timezone = pytz.timezone("America/New_York")


def get_local_time(timezone, time):
    """Convert Trality timestap to a timezone corrected time.

    Args:
        timezone (pytz.timezone): Timezone of returned time.
        time (unix timetamp): Trality timestamp.

    Returns:
        datetime: Timezone corrected datetime object.
    """

    return datetime.datetime.fromtimestamp(time / 1000.0, timezone)
    

@schedule(interval= "1d", symbol=SYMBOLS)
def handler(state, dataMap):

    for symbol, data in dataMap.items():
        today = get_local_time(state.timezone, data.times[-1])
