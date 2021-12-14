# Workaround to use Enum with Trality
# Reason: Live bots on Trality produce a pickling error for Enum.
# Fix: The TralityEnum class below returns tuples for pickling and
#   recreates the Enum at the next interval.


class TralityEnum(Enum):
    """ Base class. Inherit your Enums from this, example below.
    """
    def __getstate__(self):
        members = self.__dict__["_member_map_"]
        pairs = [(key, members[key].value) for key in members]
        return (self.__name__, pairs)

    def __setstate__(self, state):
        temp = Enum(state)
        self.__dict__.update(temp.__dict__)


class Signal(TralityEnum):
    BUY = 1  # Meets buy conditions
    SELL = 2  # Meets sell conditions
    OLD = 3  # Asset is old and not in a loss position
    IGNORE = 4  # No conditions met