""" Function to format currency and crypto for printing/logging, 
    with example of use. Crypto formatted to 6 places after the 
    decimal point. National currencies (and linked stablecoins) 
    formatter per listings at https://www.xe.com/currency/ 

    TODO: Add option to swap . for , (using locales?)
"""

SYMBOLS = ["BTC-USD", "ETH-USD"]

QUOTED_ASSET = "USD"


def initialize(state):
    pass


def format_currency(value, asset_code=QUOTED_ASSET):
    # Provide a string formatted for currencies/cryptos
    # Defaults to quoted currency/coin
    try:
        # US, Canadian, Australian Dollars and Tethered Stablecoins
        if (
            asset_code == "USD"
            or asset_code == "ZUSD"
            or asset_code == "USDC"
            or asset_code == "USDT"
            or asset_code == "UST"
            or asset_code == "USDP"
            or asset_code == "BUSD"
            or asset_code == "TUSD"
            or asset_code == "DAI"
            or asset_code == "AUD"
            or asset_code == "ZAUD"
            or asset_code == "CAD"
            or asset_code == "ZCAD"
        ):
            return_str = "${:.2f}".format(value)
        # Catchall for crypto
        elif (
            asset_code == "COIN"  # Generic
            or asset_code == "BTC"
            or asset_code == "XXBT"
            or asset_code == "ETH"
            or asset_code == "XETH"
            or asset_code == "BNB"
            or asset_code == "TRX"
            or asset_code == "VAI"
            or asset_code == "XRP"
        ):
            return_str = "{:.6f}".format(value)
        # Euro
        elif asset_code == "EURO" or asset_code == "ZEUR":
            return_str = "€{:.2f}".format(value)
        # British Pound
        elif asset_code == "GBP" or asset_code == "ZGBP":
            return_str = "£{:.2f}".format(value)
        # Swiss Franc
        elif asset_code == "CHF":
            return_str = "fr. {:.2f}".format(value)
        # Japanese Yen
        elif asset_code == "ZJPY":
            return_str = "¥{:.2f}".format(value)
        # Indonesian Rupiah Tethered Stablecoins
        elif asset_code == "BIDR" or asset_code == "IDRT":
            return_str = "RP{:.0f}".format(value)
        # Brazilian Real
        elif asset_code == "BRL":
            return_str = "R${:.2f}".format(value)
        # Vietnamese Dong Tethered Stablecoin
        elif asset_code == "BVND":
            return_str = "₫{:.2f}".format(value)
        # Nigerian Naira
        elif asset_code == "NGN":
            return_str = "₦{:.2f}".format(value)
        # Russian Ruble
        elif asset_code == "RUB":
            return_str = "₽{:.2f}".format(value)
        # Turkish Lira
        elif asset_code == "TRY":
            return_str = "₺{:.2f}".format(value)
        # Ukrainian Hryvnia
        elif asset_code == "UAH":
            return_str = "₴{:.2f}".format(value)
        else:
            return_str = str(value)
    except TypeError:
        # None or other value that cannot be formatted
        return_str = ""

    return return_str


def print_balances():
    print("{:<5} {:>13} {:>12}".format("Asset", "Amount", "Quoted_Equiv"))
    for position in query_open_positions():
        symbol_str = symbol_to_asset(position.symbol)
        print(
            "{:<5} {:>13} {:>12}".format(
                symbol_str,
                format_currency(position.exposure, "COIN"),
                format_currency(position.position_value),
            )
        )

    bal = query_balance(QUOTED_ASSET)
    print(
        "{:<5} {:>13} {:>12}".format(
            bal.asset,
            format_currency(bal.free, "COIN"),
            format_currency(bal.free),
        )
    )

    folio = query_portfolio()
    print(
        "Portfolio value: {} | {} available.".format(
            format_currency(folio.portfolio_value), format_currency(bal.free)
        )
    )


@schedule(interval="1h", symbol=SYMBOLS)
def handler(state, dataMap):

    for symbol, data in dataMap.items():
        print("{:>10} {:>10}".format(symbol, format_currency(data.close_last)))
