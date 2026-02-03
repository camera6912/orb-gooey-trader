"""Tradovate API client for automated order placement.

Supports both Demo and Live environments.
Demo: https://demo.tradovateapi.com/v1
Live: https://live.tradovateapi.com/v1

Setup:
1. In Tradovate: Settings → API Access → Generate API Key
2. Add credentials to config/secrets.yaml:
   tradovate:
     username: "your_username"
     password: "your_password"
     cid: "your_client_id"
     secret: "your_secret"
     app_id: "ORB-Gooey-Bot"
     environment: "demo"  # or "live"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Literal
import requests
from loguru import logger


@dataclass
class TradovateToken:
    """Access token with expiry tracking."""
    access_token: str
    expires_at: datetime
    user_id: int
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at - timedelta(minutes=5)


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[int] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


class TradovateClient:
    """Client for Tradovate REST API."""
    
    DEMO_URL = "https://demo.tradovateapi.com/v1"
    LIVE_URL = "https://live.tradovateapi.com/v1"
    
    # Contract IDs for common futures
    CONTRACTS = {
        "/NQ": "NQH5",  # NQ March 2025 - update as needed
        "/ES": "ESH5",  # ES March 2025
        "/MNQ": "MNQH5",  # Micro NQ
        "/MES": "MESH5",  # Micro ES
    }
    
    def __init__(
        self,
        username: str,
        password: str,
        cid: str,
        secret: str,
        app_id: str = "ORB-Gooey-Bot",
        app_version: str = "1.0",
        environment: Literal["demo", "live"] = "demo",
    ):
        self.username = username
        self.password = password
        self.cid = cid
        self.secret = secret
        self.app_id = app_id
        self.app_version = app_version
        self.environment = environment
        
        self.base_url = self.DEMO_URL if environment == "demo" else self.LIVE_URL
        self._token: Optional[TradovateToken] = None
        self._account_id: Optional[int] = None
        self._device_id: Optional[str] = None
        
        logger.info(f"TradovateClient initialized for {environment} environment")
    
    def _get_device_id(self) -> str:
        """Generate a device ID for this session."""
        if self._device_id is None:
            import uuid
            self._device_id = str(uuid.uuid4())
        return self._device_id
    
    def authenticate(self) -> bool:
        """Get access token from Tradovate."""
        if self._token and not self._token.is_expired:
            return True
        
        url = f"{self.base_url}/auth/accesstokenrequest"
        payload = {
            "name": self.username,
            "password": self.password,
            "appId": self.app_id,
            "appVersion": self.app_version,
            "cid": self.cid,
            "sec": self.secret,
            "deviceId": self._get_device_id(),
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if "accessToken" not in data:
                logger.error(f"Auth failed: {data}")
                return False
            
            # Token expires in ~8 hours typically
            expires_in = data.get("expirationTime", 28800)  # seconds
            if isinstance(expires_in, str):
                # Parse ISO timestamp
                expires_at = datetime.fromisoformat(expires_in.replace("Z", "+00:00"))
            else:
                expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            self._token = TradovateToken(
                access_token=data["accessToken"],
                expires_at=expires_at,
                user_id=data.get("userId", 0),
            )
            
            logger.info(f"Tradovate authenticated, token expires at {expires_at}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Tradovate auth error: {e}")
            return False
    
    def _headers(self) -> Dict[str, str]:
        """Get headers with auth token."""
        if not self._token:
            raise RuntimeError("Not authenticated")
        return {
            "Authorization": f"Bearer {self._token.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def get_accounts(self) -> list:
        """Get list of trading accounts."""
        if not self.authenticate():
            return []
        
        url = f"{self.base_url}/account/list"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            accounts = resp.json()
            logger.info(f"Found {len(accounts)} accounts")
            return accounts
        except requests.RequestException as e:
            logger.error(f"Failed to get accounts: {e}")
            return []
    
    def get_account_id(self) -> Optional[int]:
        """Get the primary account ID."""
        if self._account_id:
            return self._account_id
        
        accounts = self.get_accounts()
        if accounts:
            self._account_id = accounts[0].get("id")
            logger.info(f"Using account ID: {self._account_id}")
        return self._account_id
    
    def get_contract_id(self, symbol: str) -> Optional[int]:
        """Get contract ID for a symbol."""
        if not self.authenticate():
            return None
        
        # Map common symbols to contract names
        contract_name = self.CONTRACTS.get(symbol, symbol)
        
        url = f"{self.base_url}/contract/find"
        params = {"name": contract_name}
        
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
            resp.raise_for_status()
            contract = resp.json()
            return contract.get("id")
        except requests.RequestException as e:
            logger.error(f"Failed to get contract for {symbol}: {e}")
            return None
    
    def place_order(
        self,
        symbol: str,
        action: Literal["Buy", "Sell"],
        quantity: int = 1,
        order_type: Literal["Market", "Limit", "Stop", "StopLimit"] = "Market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "Day",
    ) -> OrderResult:
        """Place an order.
        
        Args:
            symbol: Contract symbol (e.g., "/NQ", "/ES")
            action: "Buy" or "Sell"
            quantity: Number of contracts
            order_type: Order type
            price: Limit price (for Limit/StopLimit orders)
            stop_price: Stop price (for Stop/StopLimit orders)
            time_in_force: "Day", "GTC", "IOC", "FOK"
        
        Returns:
            OrderResult with success status and order details
        """
        if not self.authenticate():
            return OrderResult(success=False, error="Authentication failed")
        
        account_id = self.get_account_id()
        if not account_id:
            return OrderResult(success=False, error="No account found")
        
        contract_id = self.get_contract_id(symbol)
        if not contract_id:
            return OrderResult(success=False, error=f"Contract not found: {symbol}")
        
        url = f"{self.base_url}/order/placeorder"
        payload = {
            "accountSpec": self.username,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": order_type,
            "timeInForce": time_in_force,
        }
        
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        
        try:
            logger.info(f"Placing order: {action} {quantity} {symbol} @ {order_type}")
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            order_id = data.get("orderId")
            if order_id:
                logger.info(f"Order placed successfully: {order_id}")
                return OrderResult(success=True, order_id=order_id, raw_response=data)
            else:
                error = data.get("errorText", "Unknown error")
                logger.error(f"Order failed: {error}")
                return OrderResult(success=False, error=error, raw_response=data)
                
        except requests.RequestException as e:
            logger.error(f"Order request failed: {e}")
            return OrderResult(success=False, error=str(e))
    
    def place_oco_bracket(
        self,
        symbol: str,
        entry_action: Literal["Buy", "Sell"],
        quantity: int = 1,
        entry_price: Optional[float] = None,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> OrderResult:
        """Place an OCO bracket order (entry with stop and target).
        
        For a long entry:
            - Entry: Buy
            - Stop: Sell Stop below entry
            - Target: Sell Limit above entry
        
        For a short entry:
            - Entry: Sell
            - Stop: Buy Stop above entry
            - Target: Buy Limit below entry
        """
        if not self.authenticate():
            return OrderResult(success=False, error="Authentication failed")
        
        account_id = self.get_account_id()
        if not account_id:
            return OrderResult(success=False, error="No account found")
        
        # First place entry order
        entry_result = self.place_order(
            symbol=symbol,
            action=entry_action,
            quantity=quantity,
            order_type="Market" if entry_price is None else "Limit",
            price=entry_price,
        )
        
        if not entry_result.success:
            return entry_result
        
        # Place bracket (stop + target) as OCO
        exit_action = "Sell" if entry_action == "Buy" else "Buy"
        
        url = f"{self.base_url}/order/placeoco"
        payload = {
            "accountSpec": self.username,
            "accountId": account_id,
            "action": exit_action,
            "symbol": symbol,
            "orderQty": quantity,
            "other": {
                "action": exit_action,
                "symbol": symbol,
                "orderQty": quantity,
            }
        }
        
        # Configure stop and target based on direction
        if entry_action == "Buy":  # Long position
            payload["orderType"] = "Stop"
            payload["stopPrice"] = stop_loss
            payload["other"]["orderType"] = "Limit"
            payload["other"]["price"] = take_profit
        else:  # Short position
            payload["orderType"] = "Stop"
            payload["stopPrice"] = stop_loss
            payload["other"]["orderType"] = "Limit"
            payload["other"]["price"] = take_profit
        
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"OCO bracket placed: {data}")
            return OrderResult(success=True, raw_response=data)
        except requests.RequestException as e:
            logger.error(f"OCO bracket failed: {e}")
            return OrderResult(success=False, error=str(e))
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order."""
        if not self.authenticate():
            return False
        
        url = f"{self.base_url}/order/cancelorder"
        payload = {"orderId": order_id}
        
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
            resp.raise_for_status()
            logger.info(f"Order {order_id} cancelled")
            return True
        except requests.RequestException as e:
            logger.error(f"Cancel order failed: {e}")
            return False
    
    def get_positions(self) -> list:
        """Get current positions."""
        if not self.authenticate():
            return []
        
        url = f"{self.base_url}/position/list"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Get positions failed: {e}")
            return []
    
    def close_position(self, symbol: str) -> OrderResult:
        """Close all positions for a symbol."""
        positions = self.get_positions()
        
        for pos in positions:
            if pos.get("contractId") and pos.get("netPos", 0) != 0:
                # Determine close action
                net_pos = pos["netPos"]
                action = "Sell" if net_pos > 0 else "Buy"
                qty = abs(net_pos)
                
                result = self.place_order(
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    order_type="Market",
                )
                return result
        
        return OrderResult(success=True, error="No position to close")


def load_tradovate_client(secrets_path: str = "config/secrets.yaml") -> Optional[TradovateClient]:
    """Load Tradovate client from secrets file."""
    try:
        import yaml
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f)
        
        tv = secrets.get("tradovate", {})
        if not tv.get("username"):
            logger.warning("Tradovate credentials not configured in secrets.yaml")
            return None
        
        return TradovateClient(
            username=tv["username"],
            password=tv["password"],
            cid=tv["cid"],
            secret=tv["secret"],
            app_id=tv.get("app_id", "ORB-Gooey-Bot"),
            environment=tv.get("environment", "demo"),
        )
    except Exception as e:
        logger.error(f"Failed to load Tradovate client: {e}")
        return None
