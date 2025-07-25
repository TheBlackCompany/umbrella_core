#!/usr/bin/env python3
"""
Git push helper with token authentication
"""
import os
import subprocess
import sys
from pathlib import Path

def load_token():
    """Load GitHub token from .env file"""
    env_file = Path(__file__).parent / '.env'
    if not env_file.exists():
        print("Error: .env file not found")
        print("Create .env with: GITHUB_TOKEN=your_token_here")
        return None
    
    with open(env_file) as f:
        for line in f:
            if line.startswith('GITHUB_TOKEN='):
                return line.split('=', 1)[1].strip()
    return None

def git_push(branch='main'):
    """Push to GitHub using token authentication"""
    token = load_token()
    if not token:
        return False
    
    # Get remote URL
    result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not get remote URL")
        return False
    
    remote_url = result.stdout.strip()
    
    # Extract repo info from URL
    if 'github.com' in remote_url:
        # Convert https://github.com/user/repo.git to authenticated URL
        parts = remote_url.replace('https://github.com/', '').replace('.git', '').split('/')
        if len(parts) == 2:
            user, repo = parts
            auth_url = f"https://x-access-token:{token}@github.com/{user}/{repo}.git"
            
            # Push using authenticated URL
            result = subprocess.run(['git', 'push', auth_url, branch], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully pushed to {branch}")
                return True
            else:
                print(f"Push failed: {result.stderr}")
                return False
    
    print("Error: Invalid GitHub URL format")
    return False

if __name__ == "__main__":
    branch = sys.argv[1] if len(sys.argv) > 1 else 'main'
    git_push(branch)