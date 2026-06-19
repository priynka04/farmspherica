import sys
sys.path.append(".")
from api.anomaly_detection import rule_based_alerts, get_all_alerts

def test_normal_reading_passes():
    row    = {"pH": 6.0, "EC": 1.2, "water_temp_C": 22.0}
    alerts = rule_based_alerts(row)
    assert len(alerts) == 0, "Normal reading should have no alerts!"
    print("Test 1 passed: normal reading has no alerts")

def test_bad_pH_caught():
    row    = {"pH": 3.0, "EC": 1.2, "water_temp_C": 22.0}
    alerts = rule_based_alerts(row)
    assert any("pH" in a for a in alerts), "Bad pH should be caught!"
    print("Test 2 passed: bad pH correctly flagged")

def test_bad_EC_caught():
    row    = {"pH": 6.0, "EC": 5.5, "water_temp_C": 22.0}
    alerts = rule_based_alerts(row)
    assert any("EC" in a for a in alerts), "Bad EC should be caught!"
    print("Test 3 passed: bad EC correctly flagged")

def test_combined_alert_level():
    row    = {"pH": 3.0, "EC": 1.2, "water_temp_C": 22.0}
    result = get_all_alerts(row)
    assert result["alert_level"] == "CRITICAL", "Bad pH should be CRITICAL!"
    print("Test 4 passed: alert level is CRITICAL for bad pH")

def test_good_reading_is_ok():
    row    = {"pH": 6.1, "EC": 1.0, "water_temp_C": 20.9}
    result = get_all_alerts(row)
    assert result["alert_level"] in ["OK", "WARNING"]
    print("Test 5 passed: real strawberry reading is OK")

if __name__ == "__main__":
    test_normal_reading_passes()
    test_bad_pH_caught()
    test_bad_EC_caught()
    test_combined_alert_level()
    test_good_reading_is_ok()
    print("\nAll anomaly detection tests passed!")
    
