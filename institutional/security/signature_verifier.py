"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         SIGNATURE VERIFIER - 2-Person Authentication                         ║
║                                                                               ║
║  Dual RSA signature verification for critical operations                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Implements 2-person rule for:
- Kill switch resume
- Deployment approval
- Large position changes
"""

import os
import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import base64

logger = logging.getLogger("Security.Signature")

# Try to import cryptography for RSA
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not installed - using fallback signatures")


@dataclass
class AuthorizedUser:
    """An authorized user for dual authentication."""
    user_id: str
    name: str
    public_key: Optional[bytes] = None
    role: str = "operator"  # 'operator' or 'admin'
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'name': self.name,
            'role': self.role,
            'created_at': self.created_at.isoformat(),
            'has_public_key': self.public_key is not None,
        }


@dataclass 
class SignedAction:
    """A signed action requiring approval."""
    action_id: str
    action_type: str
    payload: Dict
    created_at: datetime
    expires_at: datetime
    signatures: List[Tuple[str, str]] = field(default_factory=list)  # [(user_id, signature)]
    approved: bool = False
    executed: bool = False
    
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at


class SignatureVerifier:
    """
    Verifies RSA signatures for authenticated operations.
    
    Provides cryptographic proof that an operation was authorized.
    """
    
    def __init__(self, key_path: Optional[str] = None):
        """
        Initialize signature verifier.
        
        Args:
            key_path: Path to store/load keys
        """
        self.key_path = Path(key_path or ".keys")
        self.key_path.mkdir(parents=True, exist_ok=True)
    
    def generate_keypair(self, user_id: str) -> Tuple[bytes, bytes]:
        """
        Generate RSA keypair for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            (private_key_pem, public_key_pem)
        """
        if not CRYPTO_AVAILABLE:
            # Fallback: generate random token
            private = os.urandom(32)
            public = hashlib.sha256(private).digest()
            return private, public
        
        # Generate RSA key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Serialize
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save private key securely
        key_file = self.key_path / f"{user_id}.key"
        key_file.write_bytes(private_pem)
        try:
            os.chmod(key_file, 0o600)
        except Exception:
            pass
        
        logger.info(f"Generated keypair for user '{user_id}'")
        
        return private_pem, public_pem
    
    def sign(self, user_id: str, message: str) -> str:
        """
        Sign a message with user's private key.
        
        Args:
            user_id: User identifier
            message: Message to sign
            
        Returns:
            Base64-encoded signature
        """
        key_file = self.key_path / f"{user_id}.key"
        
        if not key_file.exists():
            raise ValueError(f"No private key for user '{user_id}'")
        
        if not CRYPTO_AVAILABLE:
            # Fallback: HMAC
            private_key = key_file.read_bytes()
            sig = hmac.new(private_key, message.encode(), hashlib.sha256).digest()
            return base64.b64encode(sig).decode()
        
        # Load private key
        private_key = serialization.load_pem_private_key(
            key_file.read_bytes(),
            password=None,
            backend=default_backend()
        )
        
        # Sign
        signature = private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def verify(self, public_key: bytes, message: str, signature: str) -> bool:
        """
        Verify a signature.
        
        Args:
            public_key: Public key bytes
            message: Original message
            signature: Base64-encoded signature
            
        Returns:
            True if valid
        """
        try:
            sig_bytes = base64.b64decode(signature)
            
            if not CRYPTO_AVAILABLE:
                # Fallback verification
                expected = hmac.new(public_key, message.encode(), hashlib.sha256).digest()
                return hmac.compare_digest(sig_bytes, expected)
            
            # Load public key
            pub_key = serialization.load_pem_public_key(
                public_key,
                backend=default_backend()
            )
            
            # Verify
            pub_key.verify(
                sig_bytes,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False


class DualAuthenticator:
    """
    Implements 2-person authentication rule.
    
    Critical operations require signatures from 2 authorized users.
    
    Usage:
        auth = DualAuthenticator()
        
        # Register users
        auth.add_user('alice', 'Alice Admin', public_key_alice)
        auth.add_user('bob', 'Bob Operator', public_key_bob)
        
        # Create action requiring approval
        action = auth.create_action('resume_trading', {'reason': 'market stable'})
        
        # Sign by first user
        auth.add_signature(action.action_id, 'alice', sig_alice)
        
        # Sign by second user
        auth.add_signature(action.action_id, 'bob', sig_bob)
        
        # Check if approved
        if auth.is_approved(action.action_id):
            execute_action()
    """
    
    def __init__(
        self,
        required_signatures: int = 2,
        action_ttl_minutes: int = 30,
        verifier: Optional[SignatureVerifier] = None,
    ):
        """
        Initialize dual authenticator.
        
        Args:
            required_signatures: Number of signatures required
            action_ttl_minutes: Time until pending action expires
            verifier: Signature verifier to use
        """
        self.required_signatures = required_signatures
        self.action_ttl_minutes = action_ttl_minutes
        self.verifier = verifier or SignatureVerifier()
        
        # Storage
        self.users: Dict[str, AuthorizedUser] = {}
        self.pending_actions: Dict[str, SignedAction] = {}
        self.completed_actions: List[SignedAction] = []
        
        # Audit
        self.audit_log: List[Dict] = []
    
    def add_user(
        self,
        user_id: str,
        name: str,
        public_key: Optional[bytes] = None,
        role: str = "operator",
    ) -> AuthorizedUser:
        """
        Add an authorized user.
        
        Args:
            user_id: Unique user ID
            name: Display name
            public_key: RSA public key (generates if None)
            role: 'operator' or 'admin'
            
        Returns:
            AuthorizedUser instance
        """
        if public_key is None:
            _, public_key = self.verifier.generate_keypair(user_id)
        
        user = AuthorizedUser(
            user_id=user_id,
            name=name,
            public_key=public_key,
            role=role,
        )
        
        self.users[user_id] = user
        self._audit("ADD_USER", user_id)
        
        logger.info(f"Added authorized user: {name} ({user_id})")
        return user
    
    def remove_user(self, user_id: str) -> bool:
        """Remove an authorized user."""
        if user_id in self.users:
            del self.users[user_id]
            self._audit("REMOVE_USER", user_id)
            return True
        return False
    
    def create_action(
        self,
        action_type: str,
        payload: Dict,
        action_id: Optional[str] = None,
    ) -> SignedAction:
        """
        Create an action requiring dual authentication.
        
        Args:
            action_type: Type of action (e.g., 'resume_trading')
            payload: Action parameters
            action_id: Custom ID (generated if None)
            
        Returns:
            SignedAction instance
        """
        if action_id is None:
            action_id = f"{action_type}_{int(time.time() * 1000)}"
        
        action = SignedAction(
            action_id=action_id,
            action_type=action_type,
            payload=payload,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=self.action_ttl_minutes),
        )
        
        self.pending_actions[action_id] = action
        self._audit("CREATE_ACTION", action_id, {"type": action_type})
        
        logger.info(f"Created action '{action_id}' requiring {self.required_signatures} signatures")
        return action
    
    def add_signature(
        self,
        action_id: str,
        user_id: str,
        signature: str,
    ) -> bool:
        """
        Add a signature to a pending action.
        
        Args:
            action_id: Action to sign
            user_id: Signing user
            signature: User's signature
            
        Returns:
            True if signature accepted
        """
        if action_id not in self.pending_actions:
            logger.warning(f"Unknown action: {action_id}")
            return False
        
        action = self.pending_actions[action_id]
        
        if action.is_expired():
            logger.warning(f"Action {action_id} has expired")
            return False
        
        if action.approved:
            logger.warning(f"Action {action_id} already approved")
            return False
        
        if user_id not in self.users:
            logger.warning(f"Unknown user: {user_id}")
            return False
        
        # Check for duplicate signature
        existing_signers = [uid for uid, _ in action.signatures]
        if user_id in existing_signers:
            logger.warning(f"User {user_id} already signed {action_id}")
            return False
        
        # Verify signature
        user = self.users[user_id]
        message = self._create_signing_message(action)
        
        if user.public_key and not self.verifier.verify(user.public_key, message, signature):
            logger.warning(f"Invalid signature from {user_id}")
            self._audit("INVALID_SIGNATURE", action_id, {"user": user_id})
            return False
        
        # Add signature
        action.signatures.append((user_id, signature))
        self._audit("ADD_SIGNATURE", action_id, {"user": user_id})
        
        logger.info(f"Signature from '{user_id}' added to {action_id} ({len(action.signatures)}/{self.required_signatures})")
        
        # Check if approved
        if len(action.signatures) >= self.required_signatures:
            action.approved = True
            self._audit("ACTION_APPROVED", action_id)
            logger.info(f"Action '{action_id}' APPROVED with {len(action.signatures)} signatures")
        
        return True
    
    def is_approved(self, action_id: str) -> bool:
        """Check if an action is approved."""
        action = self.pending_actions.get(action_id)
        if not action:
            return False
        return action.approved and not action.is_expired()
    
    def execute_action(self, action_id: str) -> Optional[Dict]:
        """
        Mark an action as executed and return its payload.
        
        Args:
            action_id: Action to execute
            
        Returns:
            Action payload if approved, None otherwise
        """
        if not self.is_approved(action_id):
            return None
        
        action = self.pending_actions.pop(action_id)
        action.executed = True
        self.completed_actions.append(action)
        
        self._audit("ACTION_EXECUTED", action_id)
        logger.info(f"Action '{action_id}' executed")
        
        return action.payload
    
    def _create_signing_message(self, action: SignedAction) -> str:
        """Create message to be signed."""
        return json.dumps({
            'action_id': action.action_id,
            'action_type': action.action_type,
            'payload': action.payload,
            'created_at': action.created_at.isoformat(),
        }, sort_keys=True)
    
    def _audit(self, event: str, target: str, details: Optional[Dict] = None) -> None:
        """Record audit entry."""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'target': target,
            'details': details or {},
        })
    
    def get_pending_actions(self) -> List[Dict]:
        """Get list of pending actions."""
        return [
            {
                'action_id': a.action_id,
                'action_type': a.action_type,
                'created_at': a.created_at.isoformat(),
                'expires_at': a.expires_at.isoformat(),
                'signatures': len(a.signatures),
                'required': self.required_signatures,
                'signers': [uid for uid, _ in a.signatures],
            }
            for a in self.pending_actions.values()
            if not a.is_expired()
        ]
    
    def get_status(self) -> Dict:
        """Get authenticator status."""
        return {
            'required_signatures': self.required_signatures,
            'authorized_users': len(self.users),
            'users': [u.to_dict() for u in self.users.values()],
            'pending_actions': len(self.pending_actions),
            'completed_actions': len(self.completed_actions),
            'audit_log_size': len(self.audit_log),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create authenticator
    auth = DualAuthenticator(required_signatures=2)
    
    # Add users
    alice = auth.add_user("alice", "Alice Admin", role="admin")
    bob = auth.add_user("bob", "Bob Operator", role="operator")
    
    print("Status:", json.dumps(auth.get_status(), indent=2))
    
    # Create action
    action = auth.create_action(
        "resume_trading",
        {"reason": "Market conditions stable"}
    )
    print(f"\nCreated action: {action.action_id}")
    
    # Sign with Alice
    sig_alice = auth.verifier.sign("alice", auth._create_signing_message(action))
    auth.add_signature(action.action_id, "alice", sig_alice)
    
    print(f"After Alice's signature: approved={auth.is_approved(action.action_id)}")
    
    # Sign with Bob
    sig_bob = auth.verifier.sign("bob", auth._create_signing_message(action))
    auth.add_signature(action.action_id, "bob", sig_bob)
    
    print(f"After Bob's signature: approved={auth.is_approved(action.action_id)}")
    
    # Execute
    payload = auth.execute_action(action.action_id)
    print(f"Executed with payload: {payload}")
    
    # Show pending
    print("\nPending actions:", auth.get_pending_actions())
