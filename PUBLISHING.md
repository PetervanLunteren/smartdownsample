# Publishing to PyPI Guide

## Package Structure
```
smartdownsample-package/
├── smartdownsample/          # Main package
│   ├── __init__.py          # Package exports
│   └── core.py              # Core functionality
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_core.py
├── examples/                # Usage examples
│   └── basic_usage.py
├── README.md               # Package documentation
├── LICENSE                 # MIT license
├── pyproject.toml         # Modern Python packaging
├── setup.py               # Fallback setup
├── MANIFEST.in            # Include additional files
└── .gitignore             # Git ignore rules
```

## Installation for Development

1. Clone/download this package
2. Install in development mode:
```bash
cd smartdownsample-package
pip install -e .
```

## Testing

Run tests to ensure everything works:
```bash
pip install pytest
pytest tests/ -v
```

## Building the Package

1. Install build tools:
```bash
pip install build twine
```

2. Build the package:
```bash
python -m build
```

This creates `dist/` folder with:
- `smartdownsample-0.1.0.tar.gz` (source distribution)
- `smartdownsample-0.1.0-py3-none-any.whl` (wheel)

## Publishing to PyPI

### Test PyPI (Recommended first)

1. Create account at https://test.pypi.org
2. Upload to test PyPI:
```bash
twine upload --repository testpypi dist/*
```
3. Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ smartdownsample
```

### Production PyPI

1. Create account at https://pypi.org
2. Upload to PyPI:
```bash
twine upload dist/*
```
3. Install from PyPI:
```bash
pip install smartdownsample
```

## GitHub Repository Setup

1. Create repository on GitHub: `smartdownsample`
2. Initialize git:
```bash
git init
git add .
git commit -m "Initial commit: SmartDownsample package"
git branch -M main
git remote add origin https://github.com/yourusername/smartdownsample.git
git push -u origin main
```

3. Update URLs in `pyproject.toml` and `setup.py` with your GitHub username

## Usage After Publishing

Once published, users can install and use:

```python
# Install
pip install smartdownsample

# Use
from smartdownsample import select_distinct

# Select 100 most diverse images
selected = select_distinct(image_paths, target_count=100)
```

## Version Updates

To release new versions:

1. Update version in `pyproject.toml` and `smartdownsample/__init__.py`
2. Build and upload new version:
```bash
python -m build
twine upload dist/*
```

## Maintenance

- Update dependencies in `pyproject.toml` as needed
- Add more examples in `examples/` folder
- Expand test suite in `tests/`
- Update README.md with new features