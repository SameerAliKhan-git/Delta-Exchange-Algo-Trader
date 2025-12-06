"""
Security - Zero-Trust Security Mesh
====================================

Comprehensive security layer including:
- HashiCorp Vault integration
- 2-person authentication
- Container signing
"""

from .vault_client import VaultClient, SecretConfig
from .signature_verifier import SignatureVerifier, DualAuthenticator

__all__ = [
    'VaultClient',
    'SecretConfig',
    'SignatureVerifier',
    'DualAuthenticator',
]

# Convenience alias
SecurityMesh = VaultClient
