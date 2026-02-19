#!/bin/bash
#
# VM Setup Script for M4 Deployment
# Run this on your GCP VM to set up Docker, Docker Compose, and SSH keys
#
# Usage: 
#   ssh root@34.134.85.31
#   curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/main/scripts/vm_setup.sh | bash
#   OR
#   ./scripts/vm_setup.sh
#

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 M4 VM Setup Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run as root"
    exit 1
fi

echo ""
echo "Step 1️⃣: Update system packages"
echo "────────────────────────────────────"
apt-get update
apt-get upgrade -y

echo ""
echo "Step 2️⃣: Install Docker"
echo "────────────────────────────────────"
if command -v docker &> /dev/null; then
    echo "✅ Docker is already installed"
    docker --version
else
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    usermod -aG docker root
    echo "✅ Docker installed successfully"
    docker --version
fi

echo ""
echo "Step 3️⃣: Install Docker Compose"
echo "────────────────────────────────────"
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is already installed"
    docker-compose --version
else
    echo "📦 Installing Docker Compose..."
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d'"' -f4)
    curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed successfully"
    docker-compose --version
fi

echo ""
echo "Step 4️⃣: Set up SSH keys for GitHub Actions"
echo "────────────────────────────────────────────"

# Ensure .ssh directory exists
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Generate SSH key if it doesn't exist
if [ ! -f ~/.ssh/github_actions ]; then
    echo "🔑 Generating SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/github_actions -N ""
else
    echo "✅ SSH key already exists"
fi

# Add public key to authorized_keys
echo "📝 Adding public key to authorized_keys..."
if ! grep -q "$(cat ~/.ssh/github_actions.pub)" ~/.ssh/authorized_keys 2>/dev/null; then
    cat ~/.ssh/github_actions.pub >> ~/.ssh/authorized_keys
    echo "✅ Public key added"
else
    echo "✅ Public key already in authorized_keys"
fi

chmod 600 ~/.ssh/authorized_keys
chmod 644 ~/.ssh/authorized_keys.pub

echo ""
echo "Step 5️⃣: Set up deployment directory"
echo "───────────────────────────────────────"
mkdir -p ~/mlops-deployment
chmod 755 ~/mlops-deployment
echo "✅ Deployment directory created at ~/mlops-deployment"

echo ""
echo "Step 6️⃣: Display SSH keys for GitHub Secrets"
echo "──────────────────────────────────────────────"
echo ""
echo "📋 Copy this and add as DEPLOY_KEY secret in GitHub:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat ~/.ssh/github_actions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo ""
echo "📋 Run this command to get SSH_HOST_KEY:"
echo "──────────────────────────────────────────"
echo "ssh-keyscan -H $(hostname -I | awk '{print $1}')"
echo ""

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ VM Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "1. Copy the private key above and add as DEPLOY_KEY in GitHub Secrets"
echo "2. Run: ssh-keyscan -H 34.134.85.31 (use your VM's IP)"
echo "3. Add SSH_HOST_KEY output to GitHub Secrets"
echo "4. Push to main branch to trigger deployment"
echo ""

exit 0
