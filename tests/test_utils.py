import pytest
import sys
sys.path.append("/home/ben/Code/nama/nama-git")
print(sys.path)
from nama.utils import simplify, simplify_corp


def test_simplify():

    # Test with all uppercase letters and punctuation
    assert simplify("HELLO, WORLD!") == "hello world"

    # Test with mixed case and special characters
    assert simplify("HeLLo, WoRlD!!!") == "hello world"

    # Test with an empty string
    assert simplify("") == ""

    # Test with a string containing only punctuation
    assert simplify("!@#$%^&*") == ""

    # Test with a string containing numbers
    assert simplify("123abc") == "123abc"

    # Test with a string containing ampersands
    assert simplify("rock & roll") == "rock and roll"

    # Test with a string containing multiple spaces and special characters
    assert simplify("  This   is    a test.    ") == "this is a test"

    # Test with a string containing different types of quotation marks
    assert simplify("‘single’ “double” ‘’´`") == "single double"

    # Test with a string containing underscores and hyphens
    assert simplify("under_score - hyphen") == "under score hyphen"

    # Test with a string containing a mix of characters
    assert simplify("Let's test this! It's 1+1=2") == "lets test this its 1+1=2"

    # Test with a string containing leading and trailing spaces
    assert simplify("  leading and trailing spaces   ") == "leading and trailing spaces"

def test_simplify_corp():
    # Test with all uppercase letters and punctuation
    assert simplify_corp("HELLO, WORLD!") == "hello world"

    # Test with mixed case and special characters
    assert simplify_corp("HeLLo, WoRlD!!!") == "hello world"

    # Test with an empty string
    assert simplify_corp("") == ""

    # Test with a string starting with 'the '
    assert simplify_corp("The Company") == "company"

    # Test with a string containing a corporate suffix
    assert simplify_corp("ABC Corp") == "abc"

    # Test with a string containing multiple corporate suffixes
    assert simplify_corp("XYZ Holding Co LLC") == "xyz"

    # Test with a string containing a corporate suffix and 'the ' prefix
    assert simplify_corp("The XYZ Holding Co") == "xyz"

    # Test with a string containing 'the ' prefix and multiple corporate suffixes
    assert simplify_corp("The ABC Holding Co LLC") == "abc"

    # Test with a string containing 'the ' prefix and multiple spaces
    assert simplify_corp("The   Corporation") == "corporation"