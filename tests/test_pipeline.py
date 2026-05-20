import pandas as pd
import sys
sys.path.append(".")
from api.data_pipeline import validate_data

def test_missing_values_detected():
    """Test that missing values are caught."""
    df = pd.DataFrame({'pH': [6.5, None, 7.0], 'temperature_C': [22, 23, 24]})
    _, issues = validate_data(df)
    assert any("Missing" in i for i in issues), "Missing values not detected!"
    print("✅ Test passed: missing values detected correctly")

def test_out_of_range_detected():
    """Test that out-of-range pH is caught."""
    df = pd.DataFrame({'pH': [6.5, 20.0, 7.0], 'temperature_C': [22, 23, 24]})
    _, issues = validate_data(df)
    assert any("pH" in i for i in issues), "Out-of-range pH not detected!"
    print("✅ Test passed: out-of-range values detected correctly")

def test_valid_data_passes():
    """Test that clean data raises no issues."""
    df = pd.DataFrame({'pH': [6.0, 6.5, 7.0], 'temperature_C': [22, 23, 24]})
    _, issues = validate_data(df)
    assert len(issues) == 0, "Clean data should have no issues!"
    print("✅ Test passed: clean data validated correctly")

if __name__ == "__main__":
    test_missing_values_detected()
    test_out_of_range_detected()
    test_valid_data_passes()
    print("\n🎉 All tests passed!")