from mapa.data.array_label import parse_array


def test_parse_array_basic_cases():
    assert parse_array("F1") == ("F", 1)
    assert parse_array("F12") == ("F", 12)
    assert parse_array("OFa1") == ("OFa", 1)
    assert parse_array("LH-3") == ("LH-", 3)


def test_parse_array_no_suffix():
    assert parse_array("noisy") == ("noisy", None)
    assert parse_array("") == ("", None)
