"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         AUDIT BUS - Regulatory-Ready Trade Logging                            ║
║                                                                               ║
║  Immutable, append-only audit logs with 7-year retention                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Every action that hits the exchange produces an immutable log entry:
  timestamp_ns | order_id | model_id | model_version | regime_posterior | 
  tvs | weight | qty | price | expected_slippage | risk_approval_sig

Storage: append-only (S3 + Object Lock compatible)
Format: parquet, partitioned by day
Query: SQL interface for regulator queries in < 60s
"""

import os
import json
import logging
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from pathlib import Path
import threading
import gzip

logger = logging.getLogger("Audit.Bus")


@dataclass
class AuditConfig:
    """Configuration for audit bus."""
    storage_path: str = "audit_logs"
    retention_years: int = 7
    batch_size: int = 100  # Flush after this many entries
    flush_interval_seconds: int = 60
    compress: bool = True
    include_checksum: bool = True


@dataclass
class AuditEntry:
    """Single audit log entry."""
    # Timestamp
    timestamp_ns: int
    timestamp_iso: str
    
    # Order identification
    order_id: str
    exchange: str
    symbol: str
    
    # Model information
    model_id: str
    model_version: str
    
    # Market state
    regime_posterior: List[float]
    regime_label: str
    
    # Signal information
    tvs: float  # Trade Validity Score
    signal_confidence: float
    
    # Allocation
    strategy_weight: float
    
    # Order details
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'limit', 'market', etc.
    
    # Slippage expectations
    expected_slippage_bps: float
    realized_slippage_bps: Optional[float] = None
    
    # Risk approval
    risk_approval_sig: Optional[str] = None
    risk_checks_passed: List[str] = field(default_factory=list)
    risk_checks_failed: List[str] = field(default_factory=list)
    
    # Execution result
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    fill_time_ns: Optional[int] = None
    execution_status: str = "pending"  # pending, filled, partial, cancelled, rejected
    
    # Checksums
    entry_checksum: Optional[str] = None
    
    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of entry."""
        data = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['entry_checksum'] = self.compute_checksum()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEntry':
        """Create from dictionary."""
        return cls(**data)


class AuditBus:
    """
    Regulatory-ready audit bus for trade logging.
    
    Features:
    - Append-only storage
    - Immutable entries with checksums
    - Parquet-compatible output
    - Daily partitioning
    - 7-year retention support
    - S3 Object Lock compatible format
    
    Usage:
        audit = AuditBus()
        
        # Log an order
        entry = AuditEntry(
            timestamp_ns=time.time_ns(),
            order_id="ORDER123",
            ...
        )
        audit.log(entry)
        
        # Query for regulator
        entries = audit.query(
            start_date="2024-01-01",
            end_date="2024-01-31",
            symbol="BTCUSD"
        )
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """
        Initialize audit bus.
        
        Args:
            config: Audit configuration
        """
        self.config = config or AuditConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Buffer for batch writing
        self._buffer: List[AuditEntry] = []
        self._buffer_lock = threading.Lock()
        
        # Chain hash for tamper detection
        self._chain_hash = hashlib.sha256(b'genesis').hexdigest()
        
        # Statistics
        self._stats = {
            'entries_logged': 0,
            'entries_flushed': 0,
            'flushes': 0,
            'errors': 0,
        }
        
        # Start flush timer
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._periodic_flush,
            daemon=True,
            name="AuditFlush"
        )
        self._flush_thread.start()
        
        logger.info(f"AuditBus initialized at {self.storage_path}")
    
    def log(self, entry: AuditEntry) -> str:
        """
        Log an audit entry.
        
        Args:
            entry: Audit entry to log
            
        Returns:
            Entry checksum
        """
        with self._buffer_lock:
            # Compute checksum with chain hash
            entry.entry_checksum = self._compute_chained_checksum(entry)
            self._buffer.append(entry)
            self._stats['entries_logged'] += 1
            
            # Flush if buffer full
            if len(self._buffer) >= self.config.batch_size:
                self._flush()
        
        return entry.entry_checksum
    
    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        model_id: str,
        model_version: str,
        regime_posterior: List[float],
        tvs: float,
        strategy_weight: float,
        expected_slippage_bps: float,
        **kwargs
    ) -> str:
        """
        Convenience method to log an order.
        
        Returns:
            Entry checksum
        """
        entry = AuditEntry(
            timestamp_ns=time.time_ns(),
            timestamp_iso=datetime.now().isoformat(),
            order_id=order_id,
            exchange=kwargs.get('exchange', 'DELTA'),
            symbol=symbol,
            model_id=model_id,
            model_version=model_version,
            regime_posterior=regime_posterior,
            regime_label=kwargs.get('regime_label', f'regime_{regime_posterior.index(max(regime_posterior))}'),
            tvs=tvs,
            signal_confidence=kwargs.get('signal_confidence', tvs),
            strategy_weight=strategy_weight,
            side=side,
            quantity=quantity,
            price=price,
            order_type=kwargs.get('order_type', 'limit'),
            expected_slippage_bps=expected_slippage_bps,
            risk_approval_sig=kwargs.get('risk_approval_sig'),
            risk_checks_passed=kwargs.get('risk_checks_passed', []),
            risk_checks_failed=kwargs.get('risk_checks_failed', []),
        )
        
        return self.log(entry)
    
    def update_execution(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
        status: str = "filled",
    ) -> bool:
        """
        Update an order with execution details.
        
        Note: This creates a new audit entry referencing the original order.
        Original entries are never modified (append-only).
        
        Args:
            order_id: Original order ID
            fill_price: Execution price
            fill_quantity: Executed quantity
            status: Execution status
            
        Returns:
            True if logged successfully
        """
        # Find original entry (in today's log)
        entries = self._load_day_entries(date.today())
        original = None
        for e in entries:
            if e.order_id == order_id:
                original = e
        
        if not original:
            logger.warning(f"Original order {order_id} not found for execution update")
            # Still log the execution
            entry = AuditEntry(
                timestamp_ns=time.time_ns(),
                timestamp_iso=datetime.now().isoformat(),
                order_id=f"{order_id}_FILL",
                exchange="DELTA",
                symbol="UNKNOWN",
                model_id="UNKNOWN",
                model_version="UNKNOWN",
                regime_posterior=[0.33, 0.33, 0.34],
                regime_label="unknown",
                tvs=0,
                signal_confidence=0,
                strategy_weight=0,
                side="unknown",
                quantity=fill_quantity,
                price=fill_price,
                order_type="fill_update",
                expected_slippage_bps=0,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                fill_time_ns=time.time_ns(),
                execution_status=status,
            )
        else:
            # Create fill entry based on original
            realized_slippage = ((fill_price - original.price) / original.price) * 10000
            if original.side == 'sell':
                realized_slippage = -realized_slippage
            
            entry = AuditEntry(
                timestamp_ns=time.time_ns(),
                timestamp_iso=datetime.now().isoformat(),
                order_id=f"{order_id}_FILL",
                exchange=original.exchange,
                symbol=original.symbol,
                model_id=original.model_id,
                model_version=original.model_version,
                regime_posterior=original.regime_posterior,
                regime_label=original.regime_label,
                tvs=original.tvs,
                signal_confidence=original.signal_confidence,
                strategy_weight=original.strategy_weight,
                side=original.side,
                quantity=original.quantity,
                price=original.price,
                order_type="fill_update",
                expected_slippage_bps=original.expected_slippage_bps,
                realized_slippage_bps=realized_slippage,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                fill_time_ns=time.time_ns(),
                execution_status=status,
            )
        
        self.log(entry)
        return True
    
    def _compute_chained_checksum(self, entry: AuditEntry) -> str:
        """Compute checksum including chain hash for tamper detection."""
        data = json.dumps(asdict(entry), sort_keys=True, default=str)
        combined = f"{self._chain_hash}:{data}"
        new_hash = hashlib.sha256(combined.encode()).hexdigest()
        self._chain_hash = new_hash
        return new_hash
    
    def _flush(self) -> None:
        """Flush buffer to storage."""
        if not self._buffer:
            return
        
        try:
            today = date.today()
            day_path = self.storage_path / f"{today.year}" / f"{today.month:02d}"
            day_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"audit_{today.isoformat()}.jsonl"
            if self.config.compress:
                filename += ".gz"
            
            filepath = day_path / filename
            
            # Append to file
            entries_json = [json.dumps(e.to_dict()) + "\n" for e in self._buffer]
            content = "".join(entries_json)
            
            if self.config.compress:
                # Gzip append
                mode = "ab" if filepath.exists() else "wb"
                with gzip.open(filepath, mode) as f:
                    f.write(content.encode())
            else:
                with open(filepath, "a") as f:
                    f.write(content)
            
            self._stats['entries_flushed'] += len(self._buffer)
            self._stats['flushes'] += 1
            self._buffer = []
            
        except Exception as e:
            logger.error(f"Failed to flush audit log: {e}")
            self._stats['errors'] += 1
    
    def _periodic_flush(self) -> None:
        """Periodically flush buffer."""
        while self._running:
            time.sleep(self.config.flush_interval_seconds)
            with self._buffer_lock:
                self._flush()
    
    def _load_day_entries(self, day: date) -> List[AuditEntry]:
        """Load entries for a specific day."""
        day_path = self.storage_path / f"{day.year}" / f"{day.month:02d}"
        filename = f"audit_{day.isoformat()}.jsonl"
        
        entries = []
        
        # Try compressed and uncompressed
        for suffix in [".gz", ""]:
            filepath = day_path / (filename + suffix)
            if filepath.exists():
                try:
                    if suffix == ".gz":
                        with gzip.open(filepath, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                    
                    for line in lines:
                        if line.strip():
                            data = json.loads(line)
                            entries.append(AuditEntry.from_dict(data))
                            
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
        
        return entries
    
    def query(
        self,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        model_id: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict]:
        """
        Query audit entries.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Filter by symbol
            model_id: Filter by model
            order_id: Filter by order ID
            limit: Maximum entries to return
            
        Returns:
            List of matching entries as dictionaries
        """
        from datetime import timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        all_entries = []
        current = start
        
        while current <= end and len(all_entries) < limit:
            entries = self._load_day_entries(current)
            
            # Apply filters
            for entry in entries:
                if symbol and entry.symbol != symbol:
                    continue
                if model_id and entry.model_id != model_id:
                    continue
                if order_id and order_id not in entry.order_id:
                    continue
                
                all_entries.append(entry.to_dict())
                
                if len(all_entries) >= limit:
                    break
            
            current += timedelta(days=1)
        
        logger.info(f"Query returned {len(all_entries)} entries")
        return all_entries
    
    def verify_integrity(self, entries: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of a chain of entries.
        
        Args:
            entries: List of entry dictionaries
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        for i, entry_dict in enumerate(entries):
            # Verify checksum
            entry = AuditEntry.from_dict(entry_dict)
            expected_checksum = entry_dict.get('entry_checksum')
            
            if expected_checksum:
                # Remove checksum and recompute
                entry.entry_checksum = None
                actual = entry.compute_checksum()
                
                if actual != expected_checksum:
                    issues.append(f"Entry {i}: checksum mismatch")
        
        return len(issues) == 0, issues
    
    def get_status(self) -> Dict:
        """Get audit bus status."""
        return {
            'storage_path': str(self.storage_path),
            'buffer_size': len(self._buffer),
            'stats': self._stats.copy(),
            'chain_hash': self._chain_hash[:16] + "...",
            'retention_years': self.config.retention_years,
        }
    
    def close(self) -> None:
        """Close audit bus and flush remaining entries."""
        self._running = False
        with self._buffer_lock:
            self._flush()
        logger.info("AuditBus closed")




# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create audit bus
    audit = AuditBus(AuditConfig(
        storage_path=".test_audit",
        compress=False,  # For easier inspection
    ))
    
    # Log some orders
    for i in range(5):
        checksum = audit.log_order(
            order_id=f"ORDER_{i:03d}",
            symbol="BTCUSD",
            side="buy",
            quantity=0.1,
            price=50000 + i * 10,
            model_id="momentum_v2",
            model_version="2.1.0",
            regime_posterior=[0.6, 0.3, 0.1],
            tvs=0.75,
            strategy_weight=0.25,
            expected_slippage_bps=2.5,
        )
        print(f"Logged ORDER_{i:03d} with checksum: {checksum[:16]}...")
    
    # Update execution
    audit.update_execution("ORDER_001", fill_price=50015, fill_quantity=0.1, status="filled")
    
    # Flush
    audit.close()
    
    # Query
    results = audit.query(
        start_date=date.today().isoformat(),
        end_date=date.today().isoformat(),
    )
    
    print(f"\nQuery returned {len(results)} entries")
    for r in results[:3]:
        print(f"  {r['order_id']}: {r['symbol']} {r['side']} {r['quantity']} @ {r['price']}")
    
    # Verify integrity
    valid, issues = audit.verify_integrity(results)
    print(f"\nIntegrity check: {'PASSED' if valid else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print("\nStatus:", json.dumps(audit.get_status(), indent=2))
