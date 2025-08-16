#!/bin/bash

# Install Git Hooks for Branch Protection
echo "Installing Git hooks for branch protection..."

# Create hooks directory
mkdir -p .git/hooks

# Install pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh

# Prevent direct commits to main branch
branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$branch" = "main" ]; then
    echo "ERROR: Direct commits to main branch are not allowed!"
    echo "Please create a feature branch first:"
    echo "   git checkout -b feature/your-feature-name"
    echo "   git add ."
    echo "   git commit -m 'your commit message'"
    echo "   git push -u origin feature/your-feature-name"
    exit 1
fi

echo "Commit allowed on branch: $branch"
exit 0
EOF

# Install pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/sh

# Prevent direct push to main branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$current_branch" = "main" ]; then
    echo "ERROR: Direct push to main branch is not allowed!"
    echo "Please use Pull Requests instead:"
    echo "   1. Create a feature branch: git checkout -b feature/your-feature"
    echo "   2. Make your changes and commit"
    echo "   3. Push feature branch: git push -u origin feature/your-feature"
    echo "   4. Create a Pull Request on GitHub"
    exit 1
fi

exit 0
EOF

# Make hooks executable
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push

echo "Git hooks installed successfully!"
echo ""
echo "Branch protection is now active:"
echo "   - Direct commits to main: BLOCKED"
echo "   - Direct push to main: BLOCKED"
echo "   - Feature branch commits: ALLOWED"
echo ""
echo "Usage:"
echo "   git checkout -b feature/my-new-feature"
echo "   # make changes..."
echo "   git add ."
echo "   git commit -m 'feat: add new feature'"
echo "   git push -u origin feature/my-new-feature"
echo ""
