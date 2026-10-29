import numpy as np

from finguard import split


def test_main_split_is_time_ordered_and_matches_config():
    day = np.arange(182)
    s = split.check(day, "main")
    assert (s["train"]["first_day"], s["train"]["last_day"]) == (0, 120)
    assert (s["valid"]["first_day"], s["valid"]["last_day"]) == (121, 151)
    assert (s["test"]["first_day"], s["test"]["last_day"]) == (152, 181)
    assert s["unused_rows"] == 0


def test_label_delay_scheme_leaves_a_30_day_gap():
    day = np.arange(182)
    s = split.check(day, "label_delay_30d")
    assert s["train"]["last_day"] == 90 and s["valid"]["first_day"] == 121
    assert s["unused_rows"] == 30
    assert (split.assign(day, "label_delay_30d")[91:121] == "unused").all()


def test_overlapping_scheme_is_rejected():
    split.SCHEMES["_bad"] = {"train": (0, 130), "valid": (121, 151), "test": (152, 181)}
    try:
        split.check(np.arange(182), "_bad")
        raised = False
    except ValueError:
        raised = True
    finally:
        del split.SCHEMES["_bad"]
    assert raised
