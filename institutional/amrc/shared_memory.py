"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         SHARED MEMORY FLAG - Ultra-Low Latency Trading Control                ║
║                                                                               ║
║  Python interface to shared memory for sub-microsecond trading halt           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

In production, this would interface with a C++ shared memory segment
that can be read by the order engine in < 5μs.
"""

import mmap
import os
import struct
import threading
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger("AMRC.SharedMemory")

# Shared memory layout:
# Byte 0: trading_enabled (1 = enabled, 0 = disabled)
# Bytes 1-8: last_update_timestamp_ns (int64)
# Bytes 9-16: halt_reason_code (int64)
SHARED_MEM_SIZE = 64
TRADING_ENABLED_OFFSET = 0
TIMESTAMP_OFFSET = 1
REASON_CODE_OFFSET = 9


class SharedMemoryFlag:
    """
    Ultra-low latency shared memory flag for trading control.
    
    This provides a mechanism for the AMRC to communicate with
    the order execution engine with minimal latency.
    
    In production, the order engine (potentially in C++) would
    mmap the same file and check the flag before every order.
    
    Usage:
        flag = SharedMemoryFlag()
        flag.set_enabled(True)
        
        # In order engine:
        if flag.is_enabled():
            execute_order()
    """
    
    def __init__(
        self,
        name: str = "amrc_trading_flag",
        path: Optional[str] = None
    ):
        """
        Initialize shared memory flag.
        
        Args:
            name: Name of the shared memory region
            path: Directory for the mmap file (default: temp)
        """
        self.name = name
        
        # Determine path
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        
        Path(path).mkdir(parents=True, exist_ok=True)
        self.filepath = os.path.join(path, f"{name}.mmap")
        
        self._lock = threading.Lock()
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the shared memory file."""
        try:
            # Create or open the file
            file_exists = os.path.exists(self.filepath)
            
            self._file = open(self.filepath, "r+b" if file_exists else "w+b")
            
            # Ensure file is correct size
            if not file_exists:
                self._file.write(b'\x01' + b'\x00' * (SHARED_MEM_SIZE - 1))
                self._file.flush()
            
            self._file.seek(0)
            
            # Create mmap
            self._mmap = mmap.mmap(
                self._file.fileno(),
                SHARED_MEM_SIZE,
                access=mmap.ACCESS_WRITE
            )
            
            logger.info(f"SharedMemoryFlag initialized at {self.filepath}")
            
        except Exception as e:
            logger.error(f"Failed to initialize shared memory: {e}")
            # Fallback to in-memory flag
            self._mmap = None
            self._fallback_flag = True
    
    def is_enabled(self) -> bool:
        """
        Check if trading is enabled.
        
        This is designed to be as fast as possible.
        Target: < 1μs
        
        Returns:
            bool: True if trading is enabled
        """
        if self._mmap is None:
            return getattr(self, '_fallback_flag', True)
        
        try:
            return self._mmap[TRADING_ENABLED_OFFSET] == 1
        except Exception:
            return getattr(self, '_fallback_flag', True)
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Set trading enabled/disabled.
        
        Args:
            enabled: Whether trading should be enabled
        """
        import time
        
        with self._lock:
            if self._mmap is None:
                self._fallback_flag = enabled
                return
            
            try:
                # Set flag
                self._mmap[TRADING_ENABLED_OFFSET] = 1 if enabled else 0
                
                # Update timestamp
                timestamp_ns = int(time.time_ns())
                timestamp_bytes = struct.pack('<Q', timestamp_ns)
                self._mmap[TIMESTAMP_OFFSET:TIMESTAMP_OFFSET + 8] = timestamp_bytes
                
                # Flush to ensure visibility
                self._mmap.flush()
                
                logger.info(f"SharedMemoryFlag set to {enabled}")
                
            except Exception as e:
                logger.error(f"Failed to set shared memory flag: {e}")
                self._fallback_flag = enabled
    
    def set_halt_reason(self, reason_code: int) -> None:
        """
        Set the halt reason code.
        
        Args:
            reason_code: Numeric reason code for the halt
        """
        with self._lock:
            if self._mmap is None:
                return
            
            try:
                reason_bytes = struct.pack('<Q', reason_code)
                self._mmap[REASON_CODE_OFFSET:REASON_CODE_OFFSET + 8] = reason_bytes
                self._mmap.flush()
            except Exception as e:
                logger.error(f"Failed to set halt reason: {e}")
    
    def get_status(self) -> dict:
        """Get current status of the shared memory flag."""
        if self._mmap is None:
            return {
                'enabled': getattr(self, '_fallback_flag', True),
                'mode': 'fallback',
                'filepath': None,
            }
        
        try:
            enabled = self._mmap[TRADING_ENABLED_OFFSET] == 1
            
            timestamp_bytes = self._mmap[TIMESTAMP_OFFSET:TIMESTAMP_OFFSET + 8]
            timestamp_ns = struct.unpack('<Q', timestamp_bytes)[0]
            
            reason_bytes = self._mmap[REASON_CODE_OFFSET:REASON_CODE_OFFSET + 8]
            reason_code = struct.unpack('<Q', reason_bytes)[0]
            
            return {
                'enabled': enabled,
                'mode': 'shared_memory',
                'filepath': self.filepath,
                'last_update_ns': timestamp_ns,
                'halt_reason_code': reason_code,
            }
        except Exception as e:
            return {
                'enabled': True,
                'mode': 'error',
                'error': str(e),
            }
    
    def close(self) -> None:
        """Close the shared memory."""
        with self._lock:
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if self._file:
                self._file.close()
                self._file = None
        
        logger.info("SharedMemoryFlag closed")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()


# Halt reason codes
class HaltReasonCode:
    """Standard halt reason codes."""
    NONE = 0
    PRICE_JUMP = 1
    API_ERRORS = 2
    BOOK_COLLAPSE = 3
    LIQUIDATION_PROXIMITY = 4
    REGIME_CONFUSION = 5
    MANUAL_HALT = 10
    SYSTEM_ERROR = 20
    NETWORK_FAILURE = 21
    EXCHANGE_MAINTENANCE = 30


# =============================================================================
# C++ HEADER GENERATION
# =============================================================================

CPP_HEADER_TEMPLATE = '''
// Auto-generated header for AMRC shared memory
// Compatible with Python SharedMemoryFlag

#ifndef AMRC_SHARED_MEMORY_H
#define AMRC_SHARED_MEMORY_H

#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>

namespace amrc {

constexpr size_t SHARED_MEM_SIZE = 64;
constexpr size_t TRADING_ENABLED_OFFSET = 0;
constexpr size_t TIMESTAMP_OFFSET = 1;
constexpr size_t REASON_CODE_OFFSET = 9;

class TradingFlag {
private:
    void* mapped_memory = nullptr;
    int fd = -1;

public:
    TradingFlag(const char* filepath) {
        fd = open(filepath, O_RDONLY);
        if (fd >= 0) {
            mapped_memory = mmap(
                nullptr, SHARED_MEM_SIZE,
                PROT_READ, MAP_SHARED,
                fd, 0
            );
        }
    }
    
    ~TradingFlag() {
        if (mapped_memory && mapped_memory != MAP_FAILED) {
            munmap(mapped_memory, SHARED_MEM_SIZE);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
    
    // Ultra-fast check - designed for < 100ns
    inline bool is_enabled() const {
        if (!mapped_memory || mapped_memory == MAP_FAILED) {
            return true;  // Fail-safe: enable trading
        }
        return *reinterpret_cast<const uint8_t*>(mapped_memory) == 1;
    }
    
    inline uint64_t get_last_update_ns() const {
        if (!mapped_memory || mapped_memory == MAP_FAILED) {
            return 0;
        }
        return *reinterpret_cast<const uint64_t*>(
            static_cast<const char*>(mapped_memory) + TIMESTAMP_OFFSET
        );
    }
    
    inline uint64_t get_halt_reason() const {
        if (!mapped_memory || mapped_memory == MAP_FAILED) {
            return 0;
        }
        return *reinterpret_cast<const uint64_t*>(
            static_cast<const char*>(mapped_memory) + REASON_CODE_OFFSET
        );
    }
};

}  // namespace amrc

#endif  // AMRC_SHARED_MEMORY_H
'''


def generate_cpp_header(output_path: str) -> None:
    """Generate C++ header for order engine integration."""
    with open(output_path, 'w') as f:
        f.write(CPP_HEADER_TEMPLATE)
    logger.info(f"Generated C++ header at {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Create flag
    flag = SharedMemoryFlag(name="test_flag")
    
    print("Initial status:", flag.get_status())
    print("Is enabled:", flag.is_enabled())
    
    # Benchmark read speed
    iterations = 100000
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = flag.is_enabled()
    elapsed_ns = time.perf_counter_ns() - start
    avg_ns = elapsed_ns / iterations
    
    print(f"\nRead benchmark: {avg_ns:.1f}ns per read ({iterations} iterations)")
    
    # Test enable/disable
    flag.set_enabled(False)
    flag.set_halt_reason(HaltReasonCode.MANUAL_HALT)
    print("\nAfter disable:", flag.get_status())
    
    flag.set_enabled(True)
    print("After enable:", flag.get_status())
    
    # Generate C++ header
    generate_cpp_header("amrc_shared_memory.h")
    print("\nGenerated C++ header")
    
    # Cleanup
    flag.close()
    print("Done")
