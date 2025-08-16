# 🛡️ Branch Protection Configuration

This document explains how to configure branch protection for this repository both locally and on GitHub.

## 🏠 Local Protection (Git Hooks)

### Automatic Installation
```bash
# Install all development tools including git hooks
make install-all

# Or install only git hooks
make install-git-hooks
```

### Manual Installation
```bash
# Run the installation script
./scripts/install_git_hooks.sh
```

### What the hooks do:
- **pre-commit**: Prevents direct commits to `main` branch
- **pre-push**: Prevents direct pushes to `main` branch

### Testing the protection:
```bash
# This will be blocked ❌
git checkout main
echo "test" > test.txt
git add test.txt
git commit -m "test"  # 🚫 ERROR: Direct commits to main branch are not allowed!

# This will work ✅
git checkout -b feature/my-feature
git add test.txt
git commit -m "feat: add test feature"  # ✅ Commit allowed
```

## 🌐 GitHub Protection (Repository Settings)

### Step 1: Access Repository Settings
1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Navigate to **Branches** in the left sidebar

### Step 2: Add Branch Protection Rule
1. Click **Add rule**
2. Branch name pattern: `main`
3. Enable the following protections:

#### Required Settings:
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: `1`
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if you have CODEOWNERS)

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - Add required status checks:
    - `pre-commit.ci` (if using pre-commit.ci)
    - `build` or `test` (if you have CI/CD)

#### Optional but Recommended:
- ✅ **Require conversation resolution before merging**
- ✅ **Restrict pushes that create files**
- ✅ **Do not allow bypassing the above settings**

#### Admin Settings:
- ❌ **Allow force pushes** (keep disabled)
- ❌ **Allow deletions** (keep disabled)

### Step 3: Save Protection Rule
Click **Create** to save the branch protection rule.

## 🔄 Recommended Workflow

### 1. Create Feature Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Make Changes
```bash
# Edit files
git add .
git commit -m "feat: add new functionality"
```

### 3. Push and Create PR
```bash
git push -u origin feature/your-feature-name
# Create Pull Request on GitHub
```

### 4. Review and Merge
- Get code review
- Ensure all checks pass
- Merge via GitHub PR interface

## 🆘 Emergency Override

If you need to bypass protections in an emergency:

### Local Override (NOT RECOMMENDED):
```bash
git commit --no-verify  # Skip pre-commit hooks
git push --no-verify    # Skip pre-push hooks
```

### GitHub Override:
- Only repository admins can force merge
- Use "Merge without waiting for requirements" (if enabled)

## 📋 Verification Checklist

After setup, verify:
- [ ] Local commits to main are blocked
- [ ] Local pushes to main are blocked  
- [ ] Feature branch commits work normally
- [ ] GitHub requires PR for main branch
- [ ] GitHub runs status checks
- [ ] PR approval is required

## 🔧 Troubleshooting

### Hook not working?
```bash
# Check if hooks exist and are executable
ls -la .git/hooks/pre-*
# Reinstall hooks
make install-git-hooks
```

### Pre-commit conflicts?
The git hooks are separate from pre-commit tool. Both can coexist:
- Git hooks: Branch protection
- Pre-commit tool: Code quality checks

### GitHub protection not working?
- Check you have admin access to repository
- Verify branch name pattern is exactly `main`
- Ensure protection rule is enabled

## 📚 References

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Git Hooks Documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
- [Pre-commit Documentation](https://pre-commit.com/)
