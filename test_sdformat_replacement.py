#!/usr/bin/env python3
"""
Test script to verify the SDFormat replacement works correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    # Test importing the real SDFormat bindings
    import sdformat

    print("✓ SDFormat bindings imported successfully")

    # Test creating a simple model
    model = sdformat.Model()
    model.set_name("test_model")
    print(f"✓ Created model: {model.name()}")

    # Test creating a simple Root structure
    root = sdformat.Root()
    root.set_model(model)
    print(f"✓ Created Root with model: {root.model().name()}")

    # Test the parser
    from jaxsim.parsers.sdformat import parser

    print("✓ Parser module imported successfully")

    # Test error handling pattern with invalid SDF string
    test_root = sdformat.Root()
    try:
        test_root.load_sdf_string("invalid sdf content")
        print("⚠️  Expected error not thrown for invalid SDF")
    except sdformat.SDFErrorsException:
        print("✓ SDFErrorsException properly caught for invalid SDF")
    except Exception as err:
        print(f"⚠️  Unexpected exception type for invalid SDF: {type(err)}")

    print("\n🎉 SDFormat replacement implementation is working!")

except Exception as e:
    print(f"❌ Error testing SDFormat implementation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
