"""
Position Manager - Track and manage positions

Provides:
- Real-time position tracking
- P&L calculation
- Position lifecycle management
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .client import DeltaClient, OrderSide


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Position information"""
    product_id: int
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    leverage: int = 1
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_open(self) -> bool:
        return self.size > 0 and self.side != PositionSide.FLAT
    
    @property
    def value(self) -> float:
        """Position notional value"""
        return self.size * self.mark_price
    
    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of entry"""
        if self.entry_price == 0 or self.margin == 0:
            return 0.0
        return (self.unrealized_pnl / self.margin) * 100


class PositionManager:
    """
    Track and manage positions
    """
    
    def __init__(
        self,
        client: DeltaClient,
        on_position_update: Callable[[Position], None] = None
    ):
        """
        Initialize position manager
        
        Args:
            client: Delta Exchange client
            on_position_update: Callback for position updates
        """
        self.client = client
        self.on_position_update = on_position_update
        
        # Track positions
        self.positions: Dict[int, Position] = {}
        self.position_history: List[Position] = []
    
    def sync_positions(self) -> None:
        """Sync positions with exchange"""
        try:
            response = self.client.get_positions()
            
            if 'result' not in response:
                return
            
            current_ids = set()
            
            for pos in response['result']:
                product_id = pos['product_id']
                current_ids.add(product_id)
                
                size = float(pos.get('size', 0))
                
                if size == 0:
                    # Position closed
                    if product_id in self.positions:
                        closed_pos = self.positions[product_id]
                        closed_pos.side = PositionSide.FLAT
                        closed_pos.size = 0
                        self.position_history.append(closed_pos)
                        del self.positions[product_id]
                    continue
                
                # Determine side
                side = PositionSide.LONG if size > 0 else PositionSide.SHORT
                
                position = Position(
                    product_id=product_id,
                    symbol=pos.get('product', {}).get('symbol', ''),
                    side=side,
                    size=abs(size),
                    entry_price=float(pos.get('entry_price', 0)),
                    mark_price=float(pos.get('mark_price', 0)),
                    liquidation_price=float(pos.get('liquidation_price', 0)),
                    leverage=int(pos.get('leverage', 1)),
                    margin=float(pos.get('margin', 0)),
                    unrealized_pnl=float(pos.get('unrealized_pnl', 0)),
                    realized_pnl=float(pos.get('realized_pnl', 0))
                )
                
                self.positions[product_id] = position
                
                # Callback
                if self.on_position_update:
                    self.on_position_update(position)
            
            # Check for closed positions
            for product_id in list(self.positions.keys()):
                if product_id not in current_ids:
                    closed_pos = self.positions[product_id]
                    closed_pos.side = PositionSide.FLAT
                    closed_pos.size = 0
                    self.position_history.append(closed_pos)
                    del self.positions[product_id]
                    
        except Exception as e:
            pass
    
    def get_position(self, product_id: int) -> Optional[Position]:
        """
        Get position for product
        
        Args:
            product_id: Product ID
        
        Returns:
            Position or None
        """
        return self.positions.get(product_id)
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Position or None
        """
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return pos
        return None
    
    def has_position(self, product_id: int) -> bool:
        """Check if position exists"""
        pos = self.positions.get(product_id)
        return pos is not None and pos.is_open
    
    def close_position(self, product_id: int) -> bool:
        """
        Close position (market order)
        
        Args:
            product_id: Product ID
        
        Returns:
            Success status
        """
        try:
            response = self.client.close_position(product_id)
            
            if 'result' in response:
                # Remove from tracking
                if product_id in self.positions:
                    closed_pos = self.positions[product_id]
                    closed_pos.side = PositionSide.FLAT
                    closed_pos.size = 0
                    self.position_history.append(closed_pos)
                    del self.positions[product_id]
                return True
            
            return False
            
        except Exception:
            return False
    
    def close_all_positions(self) -> int:
        """
        Close all positions
        
        Returns:
            Number of positions closed
        """
        closed = 0
        
        for product_id in list(self.positions.keys()):
            if self.close_position(product_id):
                closed += 1
        
        return closed
    
    def set_leverage(self, product_id: int, leverage: int) -> bool:
        """
        Set leverage for product
        
        Args:
            product_id: Product ID
            leverage: Leverage value
        
        Returns:
            Success status
        """
        try:
            response = self.client.set_leverage(product_id, leverage)
            return 'result' in response
        except Exception:
            return False
    
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def total_realized_pnl(self) -> float:
        """Get total realized P&L from history"""
        return sum(pos.realized_pnl for pos in self.position_history)
    
    def total_margin(self) -> float:
        """Get total margin in use"""
        return sum(pos.margin for pos in self.positions.values())
    
    def total_exposure(self) -> float:
        """Get total notional exposure"""
        return sum(pos.value for pos in self.positions.values())
    
    def position_summary(self) -> Dict:
        """Get position summary"""
        return {
            'open_positions': len(self.positions),
            'total_margin': self.total_margin(),
            'total_exposure': self.total_exposure(),
            'unrealized_pnl': self.total_unrealized_pnl(),
            'realized_pnl': self.total_realized_pnl(),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'size': pos.size,
                    'entry': pos.entry_price,
                    'mark': pos.mark_price,
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': pos.pnl_pct
                }
                for pos in self.positions.values()
            ]
        }
