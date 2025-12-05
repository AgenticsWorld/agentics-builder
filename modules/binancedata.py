import requests
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

BASE_URL = "https://data-api.binance.vision"

class BinanceAPI:
    """
    Binance API client for accessing Binance public market data
    """
    
    def __init__(self, timeout: int = 60):
        """
        Initialize Binance API client

        Args:
            timeout (int): API request timeout in seconds, default 60 seconds
        """
        self.base_url = BASE_URL
        # Set as tuple (connect_timeout, read_timeout), both use the same timeout value
        self.timeout = (timeout, timeout)
    
    def get_kline(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        time_zone: str = "0",
        limit: int = 500
    ) -> List[List[Union[int, str, float]]]:
        """
        Get K-line/Candlestick data
        
        Endpoint: GET /api/v3/klines
        
        Function: Get K-line data for specified trading pair, K-lines are uniquely identified by open time
        
        Parameters:
            symbol (str): Trading pair, e.g. 'ETHUSDT'
            interval (str): K-line interval, supported values:
                - Seconds: '1s'
                - Minutes: '1m', '3m', '5m', '15m', '30m'
                - Hours: '1h', '2h', '4h', '6h', '8h', '12h'
                - Days: '1d', '3d'
                - Weeks: '1w'
                - Months: '1M'
            start_time (int, optional): Start timestamp in milliseconds
            end_time (int, optional): End timestamp in milliseconds
            time_zone (str, optional): Time zone, default is "0" (UTC)
                - Supported formats: hours and minutes (e.g. '-1:00', '05:45')
                - Hours only (e.g. '0', '8', '4')
                - Accepted range: [-12:00 to +14:00]
            limit (int, optional): Number of data points to return, default 500, maximum 1000
        
        Returns:
            List[List[Union[int, str, float]]]: List of K-line data, each K-line contains the following data:
                [
                    0: int - K-line open time (milliseconds timestamp)
                    1: str - Open price
                    2: str - High price
                    3: str - Low price
                    4: str - Close price
                    5: str - Volume
                    6: int - K-line close time (milliseconds timestamp)
                    7: str - Quote asset volume
                    8: int - Number of trades
                    9: str - Taker buy base asset volume
                    10: str - Taker buy quote asset volume
                    11: str - Ignore field
                ]
        
        Notes:
            - If start_time and end_time are not provided, the most recent K-line data will be returned
            - start_time and end_time are always interpreted in UTC time, regardless of time_zone setting
        """
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "timeZone": time_zone
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception if the request fails
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Warning: Timeout while fetching {symbol} K-line data (exceeded {self.timeout[0]} seconds), skipping this request")
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"Warning: Failed to connect to Binance API ({symbol}): {str(e)}, skipping this request")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to request Binance API ({symbol}): {str(e)}, skipping this request")
            return []

    def get_price(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        symbol_status: Optional[str] = None
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Get trading pair price information

        Endpoint: GET /api/v3/ticker/price

        Function: Get the latest trading pair price information

        Parameters:
            symbol (str, optional): Single trading pair, e.g. 'BTCUSDT'
            symbols (List[str], optional): List of multiple trading pairs, e.g. ['BTCUSDT', 'BNBUSDT']
            symbol_status (str, optional): Filter by trading status, optional values:
                - 'TRADING': Trading
                - 'HALT': Halted
                - 'BREAK': Break

        Returns:
            When single trading pair, returns a dictionary:
                {
                    "symbol": "LTCBTC",
                    "price": "4.00000200"
                }

            When multiple trading pairs or all trading pairs, returns a list:
                [
                    {
                        "symbol": "LTCBTC",
                        "price": "4.00000200"
                    },
                    ...
                ]

        Weight:
            - Single trading pair: 2
            - All trading pairs (no parameters): 4
            - Multiple trading pairs: 4

        Notes:
            - symbol and symbols parameters cannot be used simultaneously
            - If neither parameter is provided, price information for all trading pairs will be returned

        Raises:
            ValueError: Raised when both symbol and symbols are provided
        """
        if symbol and symbols:
            raise ValueError("symbol and symbols parameters cannot be used simultaneously")

        endpoint = "/api/v3/ticker/price"
        params = {}

        if symbol:
            params["symbol"] = symbol
        elif symbols:
            # Convert list to JSON array format string
            import json
            params["symbols"] = json.dumps(symbols)

        if symbol_status:
            params["symbolStatus"] = symbol_status

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception if the request fails
            return response.json()
        except requests.exceptions.Timeout:
            symbol_info = symbol if symbol else "all trading pairs"
            print(f"Warning: Timeout while fetching {symbol_info} price data (exceeded {self.timeout[0]} seconds), skipping this request")
            return {} if symbol else []
        except requests.exceptions.ConnectionError as e:
            symbol_info = symbol if symbol else "all trading pairs"
            print(f"Warning: Failed to connect to Binance API ({symbol_info}): {str(e)}, skipping this request")
            return {} if symbol else []
        except requests.exceptions.RequestException as e:
            symbol_info = symbol if symbol else "all trading pairs"
            print(f"Warning: Failed to request Binance API ({symbol_info}): {str(e)}, skipping this request")
            return {} if symbol else []