#!/bin/bash
set_env() {
    if [ "$SHELL" = "/bin/fish" ]; then
        set -x "$1" "$2"
    else
        export "$1"="$2"
    fi
}

build() {
    echo "Building package..."
    echo "Ensuring Config.DEV is set to false..."
    read -p "Press Enter to continue or Ctrl+C to cancel..."

    # Clean up previous builds
    rm -rf dist/

    # Build the package
    python3 -m build
    echo "Build complete..."
}

# Function to upload to TestPyPI
push_test() {
    echo "Uploading to TestPyPI..."
    python3 -m twine upload --repository testpypi dist/* --verbose
}

# Function to upload to PyPI
push_prod() {
    echo "Uploading to PyPI..."
    python3 -m twine upload dist/* --verbose
}

# Handle command line arguments
case "$1" in
    "test")
        build
        push_test
        ;;
    "prod")
        build
        push_prod
        ;;
    "build")
        build
        ;;
    *)
        echo "Uploading to both TestPyPI and PyPI..."
        build
        push_test
        push_prod
        ;;
esac
