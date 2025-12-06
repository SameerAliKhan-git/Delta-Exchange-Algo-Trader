"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         VAULT PRODUCTION CONFIG - HashiCorp Vault Server Integration          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Production configuration for HashiCorp Vault.
This file should be used INSTEAD of local fallback in production.

Gate: vault status returns sealed=false AND no .env file exists on live pods
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger("Security.VaultProd")

# Try to import hvac (HashiCorp Vault client)
try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False
    logger.warning("hvac not installed - run: pip install hvac")


@dataclass
class VaultProdConfig:
    """Production Vault configuration."""
    # Vault server
    vault_addr: str = "https://vault.prod.internal:8200"
    
    # Authentication method
    auth_method: str = "approle"  # approle, kubernetes, token
    
    # AppRole credentials (set via environment)
    role_id: Optional[str] = None
    secret_id: Optional[str] = None
    
    # Kubernetes auth (if using k8s)
    k8s_role: str = "trading-bot"
    k8s_mount: str = "kubernetes"
    
    # Secret paths
    secrets_path: str = "secret/data/trading"
    
    # TLS
    verify_tls: bool = True
    ca_cert: Optional[str] = None
    
    # Timeouts
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'VaultProdConfig':
        """Load config from environment variables."""
        return cls(
            vault_addr=os.getenv("VAULT_ADDR", "https://vault.prod.internal:8200"),
            auth_method=os.getenv("VAULT_AUTH_METHOD", "approle"),
            role_id=os.getenv("VAULT_ROLE_ID"),
            secret_id=os.getenv("VAULT_SECRET_ID"),
            k8s_role=os.getenv("VAULT_K8S_ROLE", "trading-bot"),
            secrets_path=os.getenv("VAULT_SECRETS_PATH", "secret/data/trading"),
            verify_tls=os.getenv("VAULT_SKIP_VERIFY", "false").lower() != "true",
            ca_cert=os.getenv("VAULT_CACERT"),
        )


class VaultProdClient:
    """
    Production Vault Client using HashiCorp Vault server.
    
    This replaces the local VaultClient for production deployments.
    
    Usage:
        config = VaultProdConfig.from_env()
        vault = VaultProdClient(config)
        
        # Get API key
        api_key = vault.get_secret("delta_api_key")
    """
    
    def __init__(self, config: Optional[VaultProdConfig] = None):
        """Initialize production Vault client."""
        self.config = config or VaultProdConfig.from_env()
        self._client: Optional['hvac.Client'] = None
        self._authenticated = False
        
        # Validate environment
        self._validate_environment()
        
        # Connect
        self._connect()
    
    def _validate_environment(self) -> None:
        """Validate that we're in a production-safe environment."""
        # Gate: No .env file should exist in production
        if Path(".env").exists():
            logger.error("❌ .env file found! Remove it for production.")
            raise EnvironmentError(
                "SECURITY: .env file detected. Production requires Vault-only secrets."
            )
        
        # Warn if local fallback might be used
        if not HVAC_AVAILABLE:
            logger.error("❌ hvac library not installed. Cannot use production Vault.")
            raise ImportError(
                "SECURITY: hvac library required for production. "
                "Run: pip install hvac"
            )
    
    def _connect(self) -> None:
        """Connect to Vault server."""
        if not HVAC_AVAILABLE:
            raise ImportError("hvac library required")
        
        # Create client
        self._client = hvac.Client(
            url=self.config.vault_addr,
            verify=self.config.ca_cert if self.config.ca_cert else self.config.verify_tls,
            timeout=self.config.timeout,
        )
        
        # Authenticate
        self._authenticate()
    
    def _authenticate(self) -> None:
        """Authenticate with Vault."""
        if self.config.auth_method == "approle":
            self._auth_approle()
        elif self.config.auth_method == "kubernetes":
            self._auth_kubernetes()
        elif self.config.auth_method == "token":
            self._auth_token()
        else:
            raise ValueError(f"Unknown auth method: {self.config.auth_method}")
        
        self._authenticated = True
        logger.info(f"✅ Authenticated with Vault via {self.config.auth_method}")
    
    def _auth_approle(self) -> None:
        """Authenticate using AppRole."""
        if not self.config.role_id or not self.config.secret_id:
            raise ValueError("VAULT_ROLE_ID and VAULT_SECRET_ID required for AppRole auth")
        
        self._client.auth.approle.login(
            role_id=self.config.role_id,
            secret_id=self.config.secret_id,
        )
    
    def _auth_kubernetes(self) -> None:
        """Authenticate using Kubernetes service account."""
        # Read JWT from mounted service account
        jwt_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        
        if not Path(jwt_path).exists():
            raise ValueError(f"Kubernetes JWT not found at {jwt_path}")
        
        with open(jwt_path, 'r') as f:
            jwt = f.read()
        
        self._client.auth.kubernetes.login(
            role=self.config.k8s_role,
            jwt=jwt,
            mount_point=self.config.k8s_mount,
        )
    
    def _auth_token(self) -> None:
        """Authenticate using token (for development only)."""
        token = os.getenv("VAULT_TOKEN")
        if not token:
            raise ValueError("VAULT_TOKEN required for token auth")
        
        self._client.token = token
        
        # Warn about token auth
        logger.warning("⚠️ Using token auth - not recommended for production")
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Get a secret from Vault.
        
        Args:
            key: Secret key name
            
        Returns:
            Secret value or None
        """
        if not self._authenticated:
            raise RuntimeError("Not authenticated with Vault")
        
        try:
            secret = self._client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.config.secrets_path.split('/')[0],
            )
            
            return secret['data']['data'].get('value')
            
        except Exception as e:
            logger.error(f"Failed to read secret '{key}': {e}")
            return None
    
    def put_secret(self, key: str, value: str) -> bool:
        """
        Store a secret in Vault.
        
        Args:
            key: Secret key name
            value: Secret value
            
        Returns:
            True if successful
        """
        if not self._authenticated:
            raise RuntimeError("Not authenticated with Vault")
        
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=key,
                secret={'value': value},
                mount_point=self.config.secrets_path.split('/')[0],
            )
            
            logger.info(f"Secret '{key}' stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if Vault is healthy and unsealed."""
        try:
            status = self._client.sys.read_health_status(method='GET')
            
            # Check sealed status
            if status.get('sealed', True):
                logger.error("❌ Vault is SEALED")
                return False
            
            logger.info("✅ Vault is healthy and unsealed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vault health check failed: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get Vault status for monitoring."""
        try:
            status = self._client.sys.read_health_status(method='GET')
            
            return {
                'vault_addr': self.config.vault_addr,
                'auth_method': self.config.auth_method,
                'authenticated': self._authenticated,
                'sealed': status.get('sealed', True),
                'standby': status.get('standby', True),
                'version': status.get('version', 'unknown'),
            }
            
        except Exception as e:
            return {
                'vault_addr': self.config.vault_addr,
                'error': str(e),
                'authenticated': False,
            }


def get_vault_client(use_prod: bool = True) -> 'VaultProdClient':
    """
    Factory function to get appropriate Vault client.
    
    Args:
        use_prod: If True, use production Vault (default)
        
    Returns:
        Vault client instance
    """
    if use_prod:
        return VaultProdClient()
    else:
        # Import local fallback
        from .vault_client import VaultClient
        return VaultClient()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("VAULT PRODUCTION CLIENT TEST")
    print("=" * 60)
    
    # Check environment
    print("\n1. Environment Check:")
    print(f"   VAULT_ADDR: {os.getenv('VAULT_ADDR', '(not set)')}")
    print(f"   VAULT_AUTH_METHOD: {os.getenv('VAULT_AUTH_METHOD', '(not set)')}")
    print(f"   VAULT_ROLE_ID: {'(set)' if os.getenv('VAULT_ROLE_ID') else '(not set)'}")
    print(f"   hvac installed: {HVAC_AVAILABLE}")
    print(f"   .env exists: {Path('.env').exists()}")
    
    # Try to connect
    print("\n2. Connection Test:")
    try:
        config = VaultProdConfig.from_env()
        vault = VaultProdClient(config)
        
        print(f"   Connected to: {config.vault_addr}")
        print(f"   Healthy: {vault.is_healthy()}")
        print(f"   Status: {vault.get_status()}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print("\n   To test locally, set these environment variables:")
        print("   export VAULT_ADDR=http://127.0.0.1:8200")
        print("   export VAULT_AUTH_METHOD=token")
        print("   export VAULT_TOKEN=root")
