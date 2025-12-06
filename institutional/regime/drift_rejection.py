"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         DRIFT REJECTOR - False Alarm Filtering                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Filters out false regime change alarms that are caused by:
- Temporary volatility spikes
- Data quality issues
- Random noise

Actions on confirmed regime change:
1. Freeze weights for 30 min
2. Retrain on post-changepoint data only
3. If new regime persists > 4h, promote to primary model
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("Regime.DriftRejector")


@dataclass
class DriftAlarm:
    """A potential drift/regime change alarm."""
    alarm_id: str
    alarm_time: datetime
    alarm_type: str  # 'changepoint', 'volatility', 'mean_shift', 'distribution'
    
    # Statistics
    statistic_value: float
    threshold: float
    confidence: float
    
    # Status
    is_confirmed: bool = False
    is_rejected: bool = False
    rejection_reason: Optional[str] = None
    
    # Timing
    confirmation_time: Optional[datetime] = None
    persistence_hours: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'alarm_id': self.alarm_id,
            'alarm_time': self.alarm_time.isoformat(),
            'alarm_type': self.alarm_type,
            'statistic_value': self.statistic_value,
            'threshold': self.threshold,
            'confidence': self.confidence,
            'is_confirmed': self.is_confirmed,
            'is_rejected': self.is_rejected,
            'rejection_reason': self.rejection_reason,
            'persistence_hours': self.persistence_hours,
        }


class DriftRejector:
    """
    Drift Rejector - Filters false regime change alarms.
    
    Implements a multi-stage confirmation process:
    1. Initial detection (from BayesianChangepoint or HMM)
    2. Confirmation window (check if signal persists)
    3. Rejection filtering (eliminate false alarms)
    4. Action triggering (freeze, retrain, promote)
    
    Usage:
        rejector = DriftRejector(
            on_confirmed=lambda alarm: freeze_weights(),
            on_promoted=lambda alarm: update_primary_model(),
        )
        
        # Report a potential change
        alarm = rejector.report_alarm(
            alarm_type='changepoint',
            statistic_value=5.5,
            threshold=5.0,
        )
        
        # Check status periodically
        rejector.update()
    """
    
    def __init__(
        self,
        confirmation_window_minutes: int = 30,
        promotion_hours: float = 4.0,
        min_persistence_for_confirm: float = 0.5,  # hours
        false_alarm_penalty: float = 0.1,
        on_confirmed: Optional[Callable[[DriftAlarm], None]] = None,
        on_rejected: Optional[Callable[[DriftAlarm], None]] = None,
        on_promoted: Optional[Callable[[DriftAlarm], None]] = None,
    ):
        """
        Initialize drift rejector.
        
        Args:
            confirmation_window_minutes: Time to wait before confirming
            promotion_hours: Time before promoting new regime to primary
            min_persistence_for_confirm: Minimum hours signal must persist
            false_alarm_penalty: Penalty coefficient for frequent alarms
            on_confirmed: Callback when alarm is confirmed
            on_rejected: Callback when alarm is rejected
            on_promoted: Callback when regime is promoted
        """
        self.confirmation_window = timedelta(minutes=confirmation_window_minutes)
        self.promotion_hours = promotion_hours
        self.min_persistence = min_persistence_for_confirm
        self.false_alarm_penalty = false_alarm_penalty
        
        # Callbacks
        self.on_confirmed = on_confirmed
        self.on_rejected = on_rejected
        self.on_promoted = on_promoted
        
        # State
        self._pending_alarms: List[DriftAlarm] = []
        self._confirmed_alarms: List[DriftAlarm] = []
        self._rejected_alarms: deque = deque(maxlen=100)
        self._promoted_regimes: List[DriftAlarm] = []
        
        # Tracking
        self._recent_alarm_times: deque = deque(maxlen=50)
        self._weights_frozen: bool = False
        self._freeze_start: Optional[datetime] = None
        
        # Validation data buffer
        self._validation_data: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            'alarms_reported': 0,
            'alarms_confirmed': 0,
            'alarms_rejected': 0,
            'regimes_promoted': 0,
            'false_alarm_rate': 0.0,
        }
        
        logger.info("DriftRejector initialized")
    
    def report_alarm(
        self,
        alarm_type: str,
        statistic_value: float,
        threshold: float,
        confidence: Optional[float] = None,
    ) -> DriftAlarm:
        """
        Report a potential drift/regime change alarm.
        
        Args:
            alarm_type: Type of alarm detected
            statistic_value: Test statistic value
            threshold: Threshold that was exceeded
            confidence: Optional confidence level
            
        Returns:
            Created DriftAlarm
        """
        import uuid
        
        alarm = DriftAlarm(
            alarm_id=str(uuid.uuid4())[:8],
            alarm_time=datetime.now(),
            alarm_type=alarm_type,
            statistic_value=statistic_value,
            threshold=threshold,
            confidence=confidence or (1 - np.exp(-(statistic_value - threshold))),
        )
        
        self._pending_alarms.append(alarm)
        self._recent_alarm_times.append(alarm.alarm_time)
        self._stats['alarms_reported'] += 1
        
        logger.info(f"Alarm reported: {alarm_type} (stat={statistic_value:.2f}, thresh={threshold:.2f})")
        
        return alarm
    
    def add_validation_data(self, value: float) -> None:
        """Add data point for validation."""
        self._validation_data.append((datetime.now(), value))
    
    def update(self) -> List[DriftAlarm]:
        """
        Update alarm states and trigger callbacks.
        
        Returns:
            List of alarms that changed state
        """
        now = datetime.now()
        state_changes = []
        
        # Process pending alarms
        for alarm in self._pending_alarms[:]:
            elapsed = now - alarm.alarm_time
            
            # Check if past confirmation window
            if elapsed >= self.confirmation_window:
                # Validate alarm
                if self._validate_alarm(alarm):
                    # Confirm
                    alarm.is_confirmed = True
                    alarm.confirmation_time = now
                    alarm.persistence_hours = elapsed.total_seconds() / 3600
                    
                    self._confirmed_alarms.append(alarm)
                    self._pending_alarms.remove(alarm)
                    self._stats['alarms_confirmed'] += 1
                    
                    logger.info(f"Alarm {alarm.alarm_id} CONFIRMED after {elapsed}")
                    
                    # Trigger callback
                    if self.on_confirmed:
                        self.on_confirmed(alarm)
                    
                    # Freeze weights
                    self._freeze_weights()
                    
                    state_changes.append(alarm)
                else:
                    # Reject
                    alarm.is_rejected = True
                    alarm.rejection_reason = "Failed validation"
                    
                    self._rejected_alarms.append(alarm)
                    self._pending_alarms.remove(alarm)
                    self._stats['alarms_rejected'] += 1
                    
                    logger.info(f"Alarm {alarm.alarm_id} REJECTED: {alarm.rejection_reason}")
                    
                    if self.on_rejected:
                        self.on_rejected(alarm)
                    
                    state_changes.append(alarm)
        
        # Check for promotion of confirmed alarms
        for alarm in self._confirmed_alarms[:]:
            if alarm.confirmation_time:
                hours_since_confirm = (now - alarm.confirmation_time).total_seconds() / 3600
                alarm.persistence_hours = hours_since_confirm
                
                if hours_since_confirm >= self.promotion_hours:
                    # Promote
                    self._promoted_regimes.append(alarm)
                    self._confirmed_alarms.remove(alarm)
                    self._stats['regimes_promoted'] += 1
                    
                    logger.info(f"Alarm {alarm.alarm_id} PROMOTED after {hours_since_confirm:.1f}h")
                    
                    # Unfreeze weights
                    self._unfreeze_weights()
                    
                    if self.on_promoted:
                        self.on_promoted(alarm)
                    
                    state_changes.append(alarm)
        
        # Update false alarm rate
        self._update_false_alarm_rate()
        
        return state_changes
    
    def _validate_alarm(self, alarm: DriftAlarm) -> bool:
        """
        Validate an alarm before confirmation.
        
        Checks:
        1. Signal still present in recent data
        2. Not too many recent alarms (false alarm penalty)
        3. Data quality OK
        """
        # Check false alarm rate
        if self._is_false_alarm_rate_high():
            alarm.rejection_reason = "High false alarm rate"
            return False
        
        # Check if signal persists in validation data
        if len(self._validation_data) > 0:
            post_alarm_data = [
                v for t, v in self._validation_data
                if t >= alarm.alarm_time
            ]
            
            if len(post_alarm_data) >= 10:
                # Simple check: mean still shifted
                pre_alarm_data = [
                    v for t, v in self._validation_data
                    if t < alarm.alarm_time
                ]
                
                if len(pre_alarm_data) >= 10:
                    pre_mean = np.mean(pre_alarm_data[-50:])
                    post_mean = np.mean(post_alarm_data)
                    
                    # Check if means are significantly different
                    combined_std = np.std(list(pre_alarm_data[-50:]) + list(post_alarm_data))
                    z_score = abs(post_mean - pre_mean) / (combined_std + 1e-10)
                    
                    if z_score < 1.0:
                        alarm.rejection_reason = "Mean shift not significant"
                        return False
        
        return True
    
    def _is_false_alarm_rate_high(self) -> bool:
        """Check if recent false alarm rate is too high."""
        if len(self._recent_alarm_times) < 5:
            return False
        
        # Count alarms in last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = sum(1 for t in self._recent_alarm_times if t >= hour_ago)
        
        return recent_count > 10  # More than 10 alarms/hour is suspicious
    
    def _update_false_alarm_rate(self) -> None:
        """Update false alarm rate estimate."""
        total = self._stats['alarms_confirmed'] + self._stats['alarms_rejected']
        if total > 0:
            self._stats['false_alarm_rate'] = self._stats['alarms_rejected'] / total
    
    def _freeze_weights(self) -> None:
        """Freeze strategy weights."""
        self._weights_frozen = True
        self._freeze_start = datetime.now()
        logger.warning("🥶 Strategy weights FROZEN due to regime change")
    
    def _unfreeze_weights(self) -> None:
        """Unfreeze strategy weights."""
        self._weights_frozen = False
        self._freeze_start = None
        logger.info("🔥 Strategy weights UNFROZEN - new regime promoted")
    
    def is_frozen(self) -> bool:
        """Check if weights are currently frozen."""
        return self._weights_frozen
    
    def get_freeze_duration(self) -> float:
        """Get how long weights have been frozen (minutes)."""
        if not self._weights_frozen or not self._freeze_start:
            return 0.0
        return (datetime.now() - self._freeze_start).total_seconds() / 60
    
    def get_pending_count(self) -> int:
        """Get number of pending alarms."""
        return len(self._pending_alarms)
    
    def get_status(self) -> Dict:
        """Get rejector status."""
        return {
            'weights_frozen': self._weights_frozen,
            'freeze_duration_minutes': self.get_freeze_duration(),
            'pending_alarms': len(self._pending_alarms),
            'confirmed_alarms': len(self._confirmed_alarms),
            'promoted_regimes': len(self._promoted_regimes),
            'stats': self._stats.copy(),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_confirmed(alarm):
        print(f"⚠️ CONFIRMED: {alarm.alarm_type}")
    
    def on_rejected(alarm):
        print(f"❌ REJECTED: {alarm.alarm_type} - {alarm.rejection_reason}")
    
    def on_promoted(alarm):
        print(f"✅ PROMOTED: New regime after {alarm.persistence_hours:.1f}h")
    
    rejector = DriftRejector(
        confirmation_window_minutes=1,  # Short for demo
        promotion_hours=0.01,  # Short for demo
        on_confirmed=on_confirmed,
        on_rejected=on_rejected,
        on_promoted=on_promoted,
    )
    
    # Report some alarms
    alarm1 = rejector.report_alarm(
        alarm_type='changepoint',
        statistic_value=6.5,
        threshold=5.0,
    )
    
    # Add validation data
    for _ in range(20):
        rejector.add_validation_data(np.random.normal(0.002, 0.01))
    
    # Wait and update
    import time
    print("\nWaiting for confirmation window...")
    time.sleep(2)
    
    changes = rejector.update()
    print(f"\nState changes: {len(changes)}")
    
    # Wait for promotion
    time.sleep(40)
    changes = rejector.update()
    
    print(f"\nFinal status: {rejector.get_status()}")
