#!/usr/bin/env python3
"""
Token Generator for Clarifai Image Analysis API
Generates secure Bearer tokens for API authentication
"""

import secrets
import string
import os
from datetime import datetime

def generate_secure_token(length=32):
    """Generate a secure random token"""
    # Use a mix of letters, digits, and special characters
    alphabet = string.ascii_letters + string.digits + "-_"
    token = ''.join(secrets.choice(alphabet) for _ in range(length))
    return token

def generate_strong_token(length=48):
    """Generate a stronger token with more entropy"""
    # Use only alphanumeric for maximum compatibility
    alphabet = string.ascii_letters + string.digits
    token = ''.join(secrets.choice(alphabet) for _ in range(length))
    return token

def main():
    print("🔐 Clarifai API Token Generator")
    print("=" * 40)
    
    # Generate different token strengths
    token_32 = generate_secure_token(32)
    token_48 = generate_strong_token(48)
    token_64 = generate_strong_token(64)
    
    print(f"\n📝 Generated Tokens:")
    print(f"32 chars: {token_32}")
    print(f"48 chars: {token_48}")
    print(f"64 chars: {token_64}")
    
    # Create .env file with the recommended token
    env_content = f"""# Clarifai API Configuration
CLARIFAI_PAT=7607dc924f7d48cb9498d01f28fcb71d
CLARIFAI_USER_ID=nxi9k6mtpija
CLARIFAI_APP_ID=ScreenSnapp-Vision
CLARIFAI_MODEL_ID=set-2
CLARIFAI_MODEL_VERSION_ID=f2fb3217afa341ce87545e1ba7bf0b64

# API Configuration - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
API_BEARER_TOKEN={token_48}
"""
    
    # Write to .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Generated .env file with token: {token_48}")
    print(f"📁 File saved as: .env")
    
    print(f"\n🔑 Recommended Token (48 chars): {token_48}")
    print(f"💡 Use this token in your Authorization header:")
    print(f"   Authorization: Bearer {token_48}")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"   - Keep this token secure and private")
    print(f"   - Don't commit .env to version control")
    print(f"   - Rotate tokens regularly in production")
    
    return token_48

if __name__ == "__main__":
    main()
