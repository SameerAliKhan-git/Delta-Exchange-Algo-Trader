"""
CLI - Command Line Interface for Delta Exchange Algo Trader

Jesse-like CLI for running backtest and live trading modes.

Usage:
    python cli.py run --mode backtest --strategy momentum --symbol BTCUSD
    python cli.py run --mode live --strategy momentum --symbol BTCUSD
    python cli.py list-strategies
    python cli.py backtest --strategy momentum --start 2024-01-01 --end 2024-06-01
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_available_strategies() -> Dict[str, type]:
    """Get dictionary of available strategies"""
    from strategies import MomentumStrategy, OptionsDirectionalStrategy
    
    return {
        'momentum': MomentumStrategy,
        'options_directional': OptionsDirectionalStrategy,
    }


def run_live(args: argparse.Namespace) -> None:
    """Run live trading mode"""
    from strategies import MomentumStrategy, OptionsDirectionalStrategy
    from execution import DeltaClient, APIConfig, OrderManager, PositionManager
    from risk import RiskManager, RiskConfig
    from data import DataLoader
    
    print(f"Starting LIVE trading...")
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Testnet: {args.testnet}")
    print("-" * 50)
    
    # Load config from environment
    api_key = os.getenv('DELTA_API_KEY', '')
    api_secret = os.getenv('DELTA_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("ERROR: DELTA_API_KEY and DELTA_API_SECRET must be set")
        sys.exit(1)
    
    # Initialize client
    config = APIConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=args.testnet
    )
    client = DeltaClient(config)
    
    # Initialize managers
    order_manager = OrderManager(client)
    position_manager = PositionManager(client)
    risk_manager = RiskManager(
        RiskConfig(
            max_daily_loss_pct=args.max_loss / 100,
            max_leverage=args.leverage
        ),
        initial_capital=args.capital
    )
    
    # Get strategy class
    strategies = get_available_strategies()
    if args.strategy not in strategies:
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        print(f"Available: {', '.join(strategies.keys())}")
        sys.exit(1)
    
    strategy_class = strategies[args.strategy]
    strategy = strategy_class(
        symbol=args.symbol,
        risk_manager=risk_manager
    )
    
    # Initialize data loader
    data_loader = DataLoader(delta_client=client)
    
    print(f"Strategy initialized: {strategy.__class__.__name__}")
    print(f"Risk manager active, max daily loss: {args.max_loss}%")
    print("-" * 50)
    
    # Run main loop
    import time
    
    strategy.on_start()
    
    try:
        while True:
            # Sync positions
            position_manager.sync_positions()
            
            # Get latest candle
            candles = data_loader.load_candles(
                args.symbol,
                resolution=args.timeframe,
                limit=200
            )
            
            if len(candles) > 0:
                # Get latest candle as dict
                latest_candle = {
                    'timestamp': candles.timestamps[-1],
                    'open': candles.opens[-1],
                    'high': candles.highs[-1],
                    'low': candles.lows[-1],
                    'close': candles.closes[-1],
                    'volume': candles.volumes[-1]
                }
                
                # Call strategy
                signal = strategy.on_candle(latest_candle)
                
                if signal:
                    print(f"[{datetime.now()}] Signal: {signal}")
                    # Order execution would go here
            
            # Wait for next candle
            time.sleep(60)  # 1 minute
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        strategy.on_exit()
        print("Done.")


def run_backtest(args: argparse.Namespace) -> None:
    """Run backtest mode"""
    from strategies import MomentumStrategy, OptionsDirectionalStrategy
    from backtest import BacktestRunner, BacktestConfig, run_backtest
    from data import DataLoader, CandleData
    import numpy as np
    
    print(f"Running BACKTEST...")
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print("-" * 50)
    
    # Get strategy class
    strategies = get_available_strategies()
    if args.strategy not in strategies:
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        print(f"Available: {', '.join(strategies.keys())}")
        sys.exit(1)
    
    strategy_class = strategies[args.strategy]
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Generate sample data for demo (in production, load from API or file)
    print("Loading historical data...")
    
    # For demo, generate synthetic data
    days = (end_date - start_date).days
    n_candles = days * 24  # hourly candles
    
    timestamps = np.array([
        start_date.timestamp() + i * 3600
        for i in range(n_candles)
    ])
    
    # Generate synthetic OHLCV
    np.random.seed(42)
    base_price = 50000
    returns = np.random.randn(n_candles) * 0.01
    prices = base_price * np.cumprod(1 + returns)
    
    opens = prices
    highs = prices * (1 + np.abs(np.random.randn(n_candles) * 0.005))
    lows = prices * (1 - np.abs(np.random.randn(n_candles) * 0.005))
    closes = prices + np.random.randn(n_candles) * prices * 0.002
    volumes = np.random.uniform(100, 1000, n_candles)
    
    data = CandleData(
        symbol=args.symbol,
        resolution=args.timeframe,
        timestamps=timestamps,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes
    )
    
    print(f"Loaded {len(data)} candles")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=args.capital,
        commission_rate=args.commission / 100,
        leverage=args.leverage,
        stop_loss_pct=args.stop_loss / 100 if args.stop_loss else 0.02,
        take_profit_pct=args.take_profit / 100 if args.take_profit else 0.04
    )
    
    # Instantiate strategy
    strategy = strategy_class(symbol=args.symbol)
    
    # Run backtest
    print("Running backtest...")
    runner = BacktestRunner(strategy, config)
    result = runner.run(data)
    
    # Print results
    print(result.summary())
    
    # Save results if requested
    if args.output:
        import json
        
        output_data = {
            'strategy': args.strategy,
            'symbol': args.symbol,
            'start': args.start,
            'end': args.end,
            'initial_capital': config.initial_capital,
            'final_equity': result.final_equity,
            'total_return_pct': result.total_return_pct,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown_pct': result.metrics.max_drawdown_pct,
            'win_rate': result.metrics.win_rate,
            'total_trades': result.metrics.total_trades,
            'profit_factor': result.metrics.profit_factor
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


def list_strategies(args: argparse.Namespace) -> None:
    """List available strategies"""
    strategies = get_available_strategies()
    
    print("Available Strategies")
    print("=" * 50)
    
    for name, cls in strategies.items():
        doc = cls.__doc__ or "No description"
        doc_lines = doc.strip().split('\n')
        description = doc_lines[0] if doc_lines else "No description"
        
        print(f"\n{name}")
        print(f"  Class: {cls.__name__}")
        print(f"  Description: {description}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Delta Exchange Algo Trader - Jesse-like CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py run --mode backtest --strategy momentum --symbol BTCUSD
  python cli.py run --mode live --strategy momentum --symbol BTCUSD --testnet
  python cli.py list-strategies
  python cli.py backtest --strategy momentum -s 2024-01-01 -e 2024-06-01
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run trading strategy')
    run_parser.add_argument(
        '--mode', '-m',
        choices=['live', 'backtest', 'paper'],
        default='backtest',
        help='Trading mode'
    )
    run_parser.add_argument(
        '--strategy', '-s',
        required=True,
        help='Strategy name'
    )
    run_parser.add_argument(
        '--symbol',
        default='BTCUSD',
        help='Trading symbol'
    )
    run_parser.add_argument(
        '--timeframe', '-t',
        default='1h',
        help='Candle timeframe'
    )
    run_parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000,
        help='Initial capital'
    )
    run_parser.add_argument(
        '--leverage', '-l',
        type=int,
        default=3,
        help='Leverage'
    )
    run_parser.add_argument(
        '--max-loss',
        type=float,
        default=5.0,
        help='Max daily loss percentage'
    )
    run_parser.add_argument(
        '--testnet',
        action='store_true',
        help='Use testnet'
    )
    run_parser.add_argument(
        '--start',
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help='Backtest start date (YYYY-MM-DD)'
    )
    run_parser.add_argument(
        '--end',
        default=datetime.now().strftime("%Y-%m-%d"),
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    # Backtest command (shortcut)
    bt_parser = subparsers.add_parser('backtest', help='Run backtest (shortcut)')
    bt_parser.add_argument(
        '--strategy', '-s',
        required=True,
        help='Strategy name'
    )
    bt_parser.add_argument(
        '--symbol',
        default='BTCUSD',
        help='Trading symbol'
    )
    bt_parser.add_argument(
        '--start',
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help='Start date (YYYY-MM-DD)'
    )
    bt_parser.add_argument(
        '--end',
        default=datetime.now().strftime("%Y-%m-%d"),
        help='End date (YYYY-MM-DD)'
    )
    bt_parser.add_argument(
        '--timeframe', '-t',
        default='1h',
        help='Candle timeframe'
    )
    bt_parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000,
        help='Initial capital'
    )
    bt_parser.add_argument(
        '--leverage', '-l',
        type=int,
        default=3,
        help='Leverage'
    )
    bt_parser.add_argument(
        '--commission',
        type=float,
        default=0.06,
        help='Commission rate percentage'
    )
    bt_parser.add_argument(
        '--stop-loss',
        type=float,
        default=2.0,
        help='Stop loss percentage'
    )
    bt_parser.add_argument(
        '--take-profit',
        type=float,
        default=4.0,
        help='Take profit percentage'
    )
    bt_parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON)'
    )
    
    # List strategies command
    list_parser = subparsers.add_parser('list-strategies', help='List available strategies')
    
    # Parse args
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    if args.command == 'run':
        if args.mode == 'live':
            run_live(args)
        else:
            # For backtest mode via 'run', copy dates to args
            run_backtest(args)
    
    elif args.command == 'backtest':
        run_backtest(args)
    
    elif args.command == 'list-strategies':
        list_strategies(args)


if __name__ == '__main__':
    main()
