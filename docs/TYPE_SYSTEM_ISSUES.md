# Type System Issues in CustomerAI Project

This document outlines the current type system issues in the CustomerAI project. Last updated: May 6, 2025.

## Current Status

- Python Version: 3.10 (standardized across all components)
- Type Checking: mypy
- Linting: ruff
- Pre-commit hooks: Enabled

## Resolved Issues ✅

1. FairnessConfig circular imports
2. Missing py.typed markers
3. Undefined names in fairness_config.py
4. Import issues in __init__.py files

## Pending Issues 🔄


## 1. Ruff (Linting) Issues

### Undefined Names in fairness_config.py
- `FairnessConfig` is undefined in multiple locations:
  - Line 90: `if FairnessConfig._instance is not None` **[FIXED]**
  - Line 220: Return type annotation **[FIXED]**
  - Line 222: Function call **[FIXED]**

### Required Fixes:
- Import `FairnessConfig` from the correct module **[FIXED]**
- Ensure proper type exports in `__init__.py` files **[FIXED]**
- Fix circular import issues **[FIXED]**

## 2. MyPy (Type Checking) Issues

### Configuration Type Mismatches
```python
# src/config/fairness_config.py
- Line 177: Type mismatch in assignment **[FIXED]**
- Line 200: Type mismatch in assignment **[FIXED]**
```

### Implicit Optional Issues
```python
# src/response_generator.py
- Line 341: Implicit Optional in context parameter
# src/fairness/mitigation.py
- Line 33: Implicit Optional in config parameter
```

### Type Errors
```python
# src/model_cards.py
- Line 227: Dict get() type mismatch
```

## 3. Security Issues (Bandit)

### Subprocess Usage
Multiple instances of potentially unsafe subprocess usage in `update_project_status.py`:
- Using partial executable paths
- Not using shell=True with proper escaping
- Potential command injection vulnerabilities

## 4. Code Style Issues

### Black Formatting
- Several files need reformatting to match project style
- Inconsistent line endings (CRLF vs LF)

### Import Sorting
- Import order needs to be fixed in multiple files

## Action Items

1. **High Priority**
   - Fix circular imports in fairness configuration
   - Address all undefined name errors
   - Fix implicit Optional parameters

2. **Medium Priority**
   - Resolve type mismatches in assignments
   - Fix dict type mismatches
   - Update subprocess calls for security

3. **Low Priority**
   - Apply consistent code formatting
   - Sort imports properly
   - Fix line endings

## How to Fix

### For Developers

1. **Type Hints**
   ```python
   # Before
   def function(param = None):
       pass

   # After
   from typing import Optional
   def function(param: Optional[Dict[str, Any]] = None):
       pass
   ```

2. **Circular Imports**
   - Move type definitions to a separate types.py
   - Use string literals for forward references
   - Consider dependency injection

3. **Security**
   - Use full paths in subprocess calls
   - Properly escape user input
   - Add input validation

## Testing

After fixing these issues:
1. Run `mypy` to verify type correctness
2. Run `ruff` to check for linting issues
3. Run `black` to ensure consistent formatting
4. Run security checks with `bandit`

## Notes

- Some issues may require architectural changes
- Consider using `pyright` for additional type checking
- Document any workarounds in code comments
