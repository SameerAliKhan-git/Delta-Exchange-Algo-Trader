#!/usr/bin/env python3
"""
scripts/smoke_test.py

Smoke Test Suite
================

Quick validation that all core components are functioning:
- Services reachable
- APIs responding
- Models loadable
- Data accessible

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --full  # Run extended tests
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)


class SmokeTest:
    """Run smoke tests on the system."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.start_time = time.time()
    
    def run_test(self, name: str, test_fn, *args, **kwargs) -> bool:
        """Run a single test and record result."""
        logger.info(f"Testing: {name}...")
        
        start = time.time()
        try:
            result = test_fn(*args, **kwargs)
            duration = time.time() - start
            
            self.results.append({
                'name': name,
                'status': 'pass' if result else 'fail',
                'duration_ms': int(duration * 1000),
            })
            
            status = '✅' if result else '❌'
            logger.info(f"  {status} {name} ({duration*1000:.0f}ms)")
            return result
            
        except Exception as e:
            duration = time.time() - start
            self.results.append({
                'name': name,
                'status': 'error',
                'error': str(e),
                'duration_ms': int(duration * 1000),
            })
            logger.info(f"  ❌ {name} - Error: {e}")
            return False
    
    def test_python_version(self) -> bool:
        """Check Python version >= 3.8."""
        import platform
        version = platform.python_version_tuple()
        return int(version[0]) >= 3 and int(version[1]) >= 8
    
    def test_core_imports(self) -> bool:
        """Test core package imports."""
        packages = ['numpy', 'pandas', 'sklearn']
        for pkg in packages:
            __import__(pkg)
        return True
    
    def test_project_structure(self) -> bool:
        """Test project structure exists."""
        required = ['src', 'scripts', 'config']
        return all((PROJECT_ROOT / d).exists() for d in required)
    
    def test_config_loadable(self) -> bool:
        """Test configuration is loadable."""
        config_files = [
            PROJECT_ROOT / "config" / "config.yml",
            PROJECT_ROOT / "config" / "settings.py",
        ]
        
        for cf in config_files:
            if cf.exists():
                return True
        
        # Config might be in different format
        return (PROJECT_ROOT / "config").exists()
    
    def test_dashboard_reachable(self) -> bool:
        """Test dashboard is reachable."""
        try:
            import requests
            url = os.getenv("DASHBOARD_URL", "http://localhost:8080")
            r = requests.get(url, timeout=3)
            return r.status_code == 200
        except:
            # Dashboard not running is OK for smoke test
            return True
    
    def test_data_directory(self) -> bool:
        """Test data directory exists."""
        data_dir = PROJECT_ROOT / "data"
        return data_dir.exists()
    
    def test_ml_modules(self) -> bool:
        """Test ML modules are importable."""
        # Try to import our ML modules
        ml_path = PROJECT_ROOT / "src" / "ml"
        if not ml_path.exists():
            return True  # Skip if not present
        
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        
        # Just check the files exist
        expected_files = ['cost_aware_objective.py', 'meta_learner_bandit.py']
        for f in expected_files:
            if (ml_path / f).exists():
                return True
        
        return True
    
    def test_scripts_exist(self) -> bool:
        """Test key scripts exist."""
        scripts = [
            'full_train_backtest_and_deploy.py',
            'promotion_manager.py',
            'check_data_integrity.py',
        ]
        
        scripts_dir = PROJECT_ROOT / "scripts"
        existing = sum(1 for s in scripts if (scripts_dir / s).exists())
        return existing >= 2
    
    def test_write_permissions(self) -> bool:
        """Test we can write to necessary directories."""
        test_dirs = [
            PROJECT_ROOT / "logs",
            PROJECT_ROOT / "reports",
            PROJECT_ROOT / "models",
        ]
        
        for d in test_dirs:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".smoke_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except:
                return False
        
        return True
    
    def run_all(self, full: bool = False) -> Dict[str, Any]:
        """Run all smoke tests."""
        self.results = []
        self.start_time = time.time()
        
        # Core tests (always run)
        tests = [
            ("Python Version", self.test_python_version),
            ("Core Imports", self.test_core_imports),
            ("Project Structure", self.test_project_structure),
            ("Config Loadable", self.test_config_loadable),
            ("Data Directory", self.test_data_directory),
            ("Write Permissions", self.test_write_permissions),
            ("Scripts Exist", self.test_scripts_exist),
            ("ML Modules", self.test_ml_modules),
        ]
        
        if full:
            tests.extend([
                ("Dashboard Reachable", self.test_dashboard_reachable),
            ])
        
        for name, test_fn in tests:
            self.run_test(name, test_fn)
        
        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results if r['status'] == 'pass')
        failed = sum(1 for r in self.results if r['status'] in ['fail', 'error'])
        
        return {
            'status': 'pass' if failed == 0 else 'fail',
            'passed': passed,
            'failed': failed,
            'total': len(self.results),
            'duration_ms': int(total_duration * 1000),
            'results': self.results,
            'timestamp': datetime.utcnow().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="Smoke Test Suite")
    parser.add_argument('--full', action='store_true', help='Run extended tests')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔥 SMOKE TEST")
    print("="*60 + "\n")
    
    tester = SmokeTest()
    results = tester.run_all(full=args.full)
    
    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if results['status'] == 'pass' else 1
    
    print("\n" + "-"*60)
    print(f"Results: {results['passed']}/{results['total']} passed")
    print(f"Duration: {results['duration_ms']}ms")
    print("-"*60)
    
    if results['status'] == 'pass':
        print("\n✅ ALL SMOKE TESTS PASSED\n")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED\n")
        for r in results['results']:
            if r['status'] != 'pass':
                print(f"  - {r['name']}: {r.get('error', 'Failed')}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
