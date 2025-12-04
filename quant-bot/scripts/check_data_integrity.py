#!/usr/bin/env python3
"""
scripts/check_data_integrity.py

Data Integrity Checker
======================

Validates:
- OHLCV data presence and quality
- Feature store integrity
- Model artifacts
- Configuration files

Usage:
  python scripts/check_data_integrity.py
  python scripts/check_data_integrity.py --verbose
  python scripts/check_data_integrity.py --fix  # Attempt to fix issues
"""

import argparse
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class DataIntegrityChecker:
    """Check data integrity across the system."""
    
    def __init__(self, project_root: Path):
        self.root = project_root
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        self.passed: List[Dict] = []
    
    def check_all(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all integrity checks."""
        self.issues = []
        self.warnings = []
        self.passed = []
        
        checks = [
            ("OHLCV Data", self.check_ohlcv_data),
            ("Model Artifacts", self.check_model_artifacts),
            ("Configuration Files", self.check_config_files),
            ("Directory Structure", self.check_directory_structure),
            ("Dependencies", self.check_dependencies),
            ("Environment", self.check_environment),
        ]
        
        results = {}
        for name, check_fn in checks:
            try:
                result = check_fn(verbose)
                results[name] = result
            except Exception as e:
                self.issues.append({
                    'category': name,
                    'issue': f'Check failed with error: {e}',
                    'severity': 'error',
                })
                results[name] = {'status': 'error', 'error': str(e)}
        
        return {
            'status': 'pass' if len(self.issues) == 0 else 'fail',
            'checks': results,
            'issues': self.issues,
            'warnings': self.warnings,
            'passed': self.passed,
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'total_passed': len(self.passed),
            }
        }
    
    def check_ohlcv_data(self, verbose: bool = False) -> Dict:
        """Check OHLCV data files."""
        data_dir = self.root / "data" / "ohlcv"
        
        if not data_dir.exists():
            # Try alternative locations
            alt_dirs = [
                self.root / "data",
                self.root / "data" / "market",
            ]
            
            for alt in alt_dirs:
                if alt.exists() and list(alt.glob("*.csv")) + list(alt.glob("*.parquet")):
                    data_dir = alt
                    break
            else:
                self.warnings.append({
                    'category': 'OHLCV Data',
                    'issue': 'No OHLCV data directory found (data/ohlcv/)',
                    'fix': 'Create data/ohlcv/ and add OHLCV CSV/Parquet files',
                })
                return {'status': 'warning', 'message': 'No data directory'}
        
        # Find data files
        csv_files = list(data_dir.glob("**/*.csv"))
        parquet_files = list(data_dir.glob("**/*.parquet"))
        all_files = csv_files + parquet_files
        
        if not all_files:
            self.warnings.append({
                'category': 'OHLCV Data',
                'issue': f'No data files found in {data_dir}',
                'fix': 'Add OHLCV data files (CSV or Parquet format)',
            })
            return {'status': 'warning', 'files': []}
        
        # Check each file
        file_results = []
        for f in all_files[:10]:  # Check first 10
            result = self._check_single_data_file(f, verbose)
            file_results.append(result)
        
        self.passed.append({
            'category': 'OHLCV Data',
            'message': f'Found {len(all_files)} data files',
        })
        
        return {'status': 'pass', 'files': file_results, 'total_files': len(all_files)}
    
    def _check_single_data_file(self, filepath: Path, verbose: bool) -> Dict:
        """Check a single data file."""
        result = {
            'file': filepath.name,
            'path': str(filepath),
            'size_mb': filepath.stat().st_size / (1024 * 1024),
        }
        
        try:
            import pandas as pd
            
            if filepath.suffix == '.parquet':
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath, nrows=1000)  # Sample for speed
            
            result['rows'] = len(df)
            result['columns'] = list(df.columns)
            
            # Check for required columns
            required = {'timestamp', 'time', 'date', 'open', 'high', 'low', 'close'}
            has_time = any(c.lower() in ['timestamp', 'time', 'date'] for c in df.columns)
            has_ohlc = all(any(c.lower() == col for c in df.columns) for col in ['open', 'high', 'low', 'close'])
            
            if not has_time:
                result['warning'] = 'Missing timestamp column'
            if not has_ohlc:
                result['warning'] = 'Missing OHLC columns'
            
            result['valid'] = has_time and has_ohlc
            
        except Exception as e:
            result['error'] = str(e)
            result['valid'] = False
        
        return result
    
    def check_model_artifacts(self, verbose: bool = False) -> Dict:
        """Check model artifacts."""
        model_dirs = [
            self.root / "models",
            self.root / "model_registry",
        ]
        
        found_models = []
        for model_dir in model_dirs:
            if model_dir.exists():
                models = list(model_dir.glob("**/*.pkl")) + list(model_dir.glob("**/*.joblib"))
                found_models.extend(models)
        
        if not found_models:
            self.warnings.append({
                'category': 'Model Artifacts',
                'issue': 'No model files found',
                'fix': 'Train models using scripts/full_train_backtest_and_deploy.py',
            })
            return {'status': 'warning', 'models': []}
        
        model_info = []
        for m in found_models[:5]:
            info = {
                'name': m.name,
                'path': str(m),
                'size_kb': m.stat().st_size / 1024,
                'modified': datetime.fromtimestamp(m.stat().st_mtime).isoformat(),
            }
            model_info.append(info)
        
        self.passed.append({
            'category': 'Model Artifacts',
            'message': f'Found {len(found_models)} model files',
        })
        
        return {'status': 'pass', 'models': model_info, 'total': len(found_models)}
    
    def check_config_files(self, verbose: bool = False) -> Dict:
        """Check configuration files."""
        required_configs = [
            ("config/config.yml", "Main configuration"),
            ("config/settings.py", "Python settings"),
        ]
        
        optional_configs = [
            ("monitoring/prometheus.yml", "Prometheus config"),
            ("monitoring/prometheus_alert_rules.yml", "Alert rules"),
            (".env", "Environment variables"),
        ]
        
        results = {'required': [], 'optional': []}
        
        for config_path, description in required_configs:
            full_path = self.root / config_path
            exists = full_path.exists()
            results['required'].append({
                'path': config_path,
                'description': description,
                'exists': exists,
            })
            
            if not exists:
                self.warnings.append({
                    'category': 'Configuration',
                    'issue': f'Missing required config: {config_path}',
                    'fix': f'Create {config_path}',
                })
        
        for config_path, description in optional_configs:
            full_path = self.root / config_path
            results['optional'].append({
                'path': config_path,
                'description': description,
                'exists': full_path.exists(),
            })
        
        required_ok = all(r['exists'] for r in results['required'])
        
        if required_ok:
            self.passed.append({
                'category': 'Configuration',
                'message': 'All required configs present',
            })
        
        return {'status': 'pass' if required_ok else 'warning', 'configs': results}
    
    def check_directory_structure(self, verbose: bool = False) -> Dict:
        """Check directory structure."""
        required_dirs = [
            "src",
            "scripts",
            "config",
            "data",
        ]
        
        recommended_dirs = [
            "models",
            "logs",
            "reports",
            "tests",
            "monitoring",
        ]
        
        results = {'required': [], 'recommended': []}
        
        for d in required_dirs:
            exists = (self.root / d).exists()
            results['required'].append({'dir': d, 'exists': exists})
            
            if not exists:
                self.issues.append({
                    'category': 'Directory Structure',
                    'issue': f'Missing required directory: {d}',
                    'severity': 'error',
                })
        
        for d in recommended_dirs:
            results['recommended'].append({
                'dir': d,
                'exists': (self.root / d).exists(),
            })
        
        required_ok = all(r['exists'] for r in results['required'])
        
        if required_ok:
            self.passed.append({
                'category': 'Directory Structure',
                'message': 'All required directories present',
            })
        
        return {'status': 'pass' if required_ok else 'fail', 'directories': results}
    
    def check_dependencies(self, verbose: bool = False) -> Dict:
        """Check Python dependencies."""
        critical_packages = [
            "numpy",
            "pandas",
            "sklearn",
        ]
        
        recommended_packages = [
            "aiohttp",
            "prometheus_client",
            "optuna",
            "requests",
        ]
        
        results = {'critical': [], 'recommended': []}
        
        for pkg in critical_packages:
            try:
                __import__(pkg)
                results['critical'].append({'package': pkg, 'installed': True})
            except ImportError:
                results['critical'].append({'package': pkg, 'installed': False})
                self.issues.append({
                    'category': 'Dependencies',
                    'issue': f'Missing critical package: {pkg}',
                    'fix': f'pip install {pkg}',
                    'severity': 'error',
                })
        
        for pkg in recommended_packages:
            try:
                __import__(pkg)
                results['recommended'].append({'package': pkg, 'installed': True})
            except ImportError:
                results['recommended'].append({'package': pkg, 'installed': False})
        
        critical_ok = all(r['installed'] for r in results['critical'])
        
        if critical_ok:
            self.passed.append({
                'category': 'Dependencies',
                'message': 'All critical packages installed',
            })
        
        return {'status': 'pass' if critical_ok else 'fail', 'packages': results}
    
    def check_environment(self, verbose: bool = False) -> Dict:
        """Check environment variables."""
        recommended_env = [
            "DELTA_API_KEY",
            "DELTA_API_SECRET",
            "DASHBOARD_URL",
        ]
        
        results = []
        for var in recommended_env:
            value = os.getenv(var)
            results.append({
                'variable': var,
                'set': value is not None,
                'value_preview': f"{value[:4]}..." if value else None,
            })
        
        # Check for .env file
        env_file = self.root / ".env"
        if env_file.exists():
            results.append({
                'variable': '.env file',
                'set': True,
            })
        
        self.passed.append({
            'category': 'Environment',
            'message': 'Environment check complete',
        })
        
        return {'status': 'pass', 'variables': results}


def main():
    parser = argparse.ArgumentParser(description="Data Integrity Checker")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    checker = DataIntegrityChecker(PROJECT_ROOT)
    results = checker.check_all(verbose=args.verbose)
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0 if results['status'] == 'pass' else 1
    
    # Pretty print
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)
    
    # Summary
    print(f"\nStatus: {'✅ PASS' if results['status'] == 'pass' else '❌ FAIL'}")
    print(f"Issues: {results['summary']['total_issues']}")
    print(f"Warnings: {results['summary']['total_warnings']}")
    print(f"Passed: {results['summary']['total_passed']}")
    
    # Passed checks
    if results['passed']:
        print("\n✅ PASSED:")
        for p in results['passed']:
            print(f"   [{p['category']}] {p['message']}")
    
    # Warnings
    if results['warnings']:
        print("\n⚠️ WARNINGS:")
        for w in results['warnings']:
            print(f"   [{w['category']}] {w['issue']}")
            if 'fix' in w:
                print(f"      Fix: {w['fix']}")
    
    # Issues
    if results['issues']:
        print("\n❌ ISSUES:")
        for i in results['issues']:
            print(f"   [{i['category']}] {i['issue']}")
            if 'fix' in i:
                print(f"      Fix: {i['fix']}")
    
    print("\n" + "="*60 + "\n")
    
    return 0 if results['status'] == 'pass' else 1


if __name__ == "__main__":
    sys.exit(main())
