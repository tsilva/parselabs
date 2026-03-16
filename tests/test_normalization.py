from parselabs.normalization import preprocess_numeric_value


def test_preprocess_numeric_value_collapses_decimal_spacing():
    assert preprocess_numeric_value("13 .0") == "13.0"
    assert preprocess_numeric_value("4 ,04") == "4.04"
    assert preprocess_numeric_value(" 0 , 60 ") == "0.60"
