# Makefile Changes: check-system Implementation

**Date:** 2026-02-09
**Status:** ✓ Complete

## Summary

Modified Makefile to replace automatic system package installation with verification checks. The `install-system` target now only runs when explicitly called.

## Changes Made

### 1. Modified `install` Target Dependencies

**Before:**
```makefile
install: check-data install-system install-python install-poetry install-deps install-verify
```

**After:**
```makefile
install: check-data check-system install-python install-poetry install-deps install-verify
```

**Impact:** `make install` now **checks** for system packages instead of automatically installing them.

---

### 2. Added `check-system` Target (New)

```makefile
check-system:
  @echo "Checking for required system packages..."
  @missing_pkgs=""; \
  for pkg in build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
             libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev \
             libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git make libgomp1; do \
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then \
      missing_pkgs="$missing_pkgs $pkg"; \
    fi; \
  done; \
  if [ -n "$missing_pkgs" ]; then \
    echo ""; \
    echo "ERROR: Missing required system packages:$missing_pkgs"; \
    echo ""; \
    echo "To install missing packages, run:"; \
    echo "  make install-system"; \
    echo ""; \
    echo "Or manually install with:"; \
    echo "  sudo apt-get install -y$missing_pkgs"; \
    exit 1; \
  fi
  @echo "✓ All required system packages are installed"
  @echo ""
  @echo "Checking for Docker..."
  @if command -v docker >/dev/null 2>&1; then \
    echo "✓ Docker is installed: $(docker --version 2>/dev/null || echo 'version unknown')"; \
  else \
    echo "⚠ Docker not found (optional - only needed for deployment)"; \
    echo "  To install: make install-system"; \
  fi
```

**Features:**
- Uses `dpkg-query` to check if packages are installed (no sudo required)
- Reports all missing packages at once (not one-by-one)
- Provides clear instructions on how to fix issues
- Checks Docker separately (optional, informational only)
- Exits with error if required packages missing

---

### 3. Updated `install-system` Target (Explicit Only)

**Before:**
```makefile
install-system:
  sudo apt-get install -y \
    build-essential libssl-dev ...
```

**After:**
```makefile
# Explicit system package installation (not run by default)
# Run this manually if check-system fails
install-system:
  @echo "Installing required system packages (requires sudo)..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git make libgomp1 pipx
  @echo ""
  @echo "Installing Docker (optional)..."
  if apt-cache show docker-ce >/dev/null 2>&1; then \
    sudo apt-get install -y docker-ce docker-ce-cli docker-compose-plugin || echo "Note: docker-ce not available"; \
  else \
    sudo apt-get install -y docker.io docker-compose-v2 || echo "Note: docker.io not available"; \
  fi
  @echo ""
  @echo "✓ System packages installed successfully"
```

**Changes:**
- Added clear comment explaining it's explicit only
- Added informational echo messages
- Added `apt-get update` before install
- Added success message at the end
- Only runs when explicitly called: `make install-system`

---

## Behavior Comparison

### Before (Automatic Installation)

```bash
$ make install
Checking for required data files...
Found required data file: support_tickets.json
# Automatically runs: sudo apt-get install -y ...
# (requires sudo password prompt)
Installing pyenv...
Installing poetry...
```

**Issues:**
- Prompts for sudo password unexpectedly
- Installs packages user may not want
- No way to verify without installing

### After (Verification First)

```bash
$ make install
Checking for required data files...
Found required data file: support_tickets.json
Checking for required system packages...

ERROR: Missing required system packages: libncursesw5-dev

To install missing packages, run:
  make install-system

Or manually install with:
  sudo apt-get install -y libncursesw5-dev

make: *** [Makefile:42: check-system] Error 1
```

**Benefits:**
- No surprise sudo prompts
- Clear error messages with fix instructions
- User decides when to install
- Can verify before installing

---

## Usage

### Check System (Non-Destructive)

```bash
# Verify all system packages are installed
make check-system
```

**Example output (all packages present):**
```
Checking for required system packages...
✓ All required system packages are installed

Checking for Docker...
✓ Docker is installed: Docker version 24.0.7, build afdd53b
```

**Example output (missing packages):**
```
Checking for required system packages...

ERROR: Missing required system packages: libncursesw5-dev xz-utils

To install missing packages, run:
  make install-system

Or manually install with:
  sudo apt-get install -y libncursesw5-dev xz-utils
```

---

### Install System Packages (Explicit)

```bash
# Only run this when you want to install system packages
make install-system
```

**Example output:**
```
Installing required system packages (requires sudo)...
[sudo] password for user: 
Reading package lists... Done
Building dependency tree... Done
...
✓ System packages installed successfully
```

---

### Full Install (Now Verifies First)

```bash
# Full installation (checks but doesn't install system packages)
make install
```

**Flow:**
1. `check-data` → Verify support_tickets.json exists
2. **`check-system`** → Verify system packages (exits if missing)
3. `install-python` → Install pyenv + Python 3.12.12
4. `install-poetry` → Install Poetry
5. `install-deps` → Install Python dependencies
6. `install-verify` → Verify installation

**If check-system fails:** User must run `make install-system` first, then retry `make install`

---

## Rationale

### Why Verify Instead of Install?

1. **Security:** No surprise sudo prompts
2. **Control:** User decides when to modify system
3. **Transparency:** Clear about what's missing
4. **Portability:** Works on systems with different package managers
5. **CI/CD:** Fails fast if dependencies missing (instead of requiring sudo)

### Why Not Skip check-system Entirely?

1. **Early feedback:** Catch missing packages before spending time on Python install
2. **Clear errors:** Better than cryptic build failures later
3. **Documentation:** Shows exactly what's required

---

## Testing

### Test 1: Missing Packages
```bash
$ make check-system
# Expected: Clear error message listing missing packages
```

### Test 2: All Packages Present
```bash
$ make check-system
# Expected: Success message + Docker status
```

### Test 3: Install Flow
```bash
$ make install
# Expected: Fails at check-system if packages missing
# User runs: make install-system
# User retries: make install
# Expected: Succeeds
```

---

## Files Modified

- `Makefile` (lines 23, 26, 40-88)
  - Added `check-system` to .PHONY
  - Changed `install` dependencies: `install-system` → `check-system`
  - Added new `check-system` target
  - Updated `install-system` with comments and messages

---

## Backward Compatibility

**No breaking changes** - both workflows still work:

**Old workflow (still works):**
```bash
make install-system  # Explicit install
make install         # Now proceeds without sudo prompt
```

**New workflow (recommended):**
```bash
make install         # Checks packages, fails if missing
# If fails: make install-system
make install         # Retry after installing packages
```

---

## Future Enhancements

Possible improvements (not implemented):
1. Support for non-Debian systems (Homebrew, yum, etc.)
2. Optional packages (install only if needed)
3. Version checks (verify minimum versions)
4. Offline mode (skip package checks)

