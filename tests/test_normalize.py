import pandas as pd

from finguard.features import normalize as N


def test_browser_family_and_major():
    s = pd.Series(["chrome 63.0", "mobile safari 11.0", "samsung browser 6.2", "ie 11.0 for desktop",
                   "chrome 67.0 for android", "edge 15.0", "firefox 57.0", "opera", "weird thing", None])
    fam = N.browser_family(s).tolist()
    assert fam[:9] == ["chrome", "safari", "samsung", "internet explorer", "chrome", "edge", "firefox", "opera", "other"]
    assert pd.isna(fam[9])
    major = N.browser_major(s)
    assert major.iloc[0] == 63 and major.iloc[1] == 11 and pd.isna(major.iloc[7])


def test_os_family():
    s = pd.Series(["Windows 10", "iOS 11.2.1", "Mac OS X 10_13_2", "Android 7.0", "Linux", "func", None])
    out = N.os_family(s).tolist()
    assert out[:6] == ["windows", "ios", "mac", "android", "linux", "other"]
    assert pd.isna(out[6])


def test_device_brand():
    s = pd.Series(["SAMSUNG SM-G892A Build/NRD90M", "iOS Device", "Windows", "MacOS", "Moto G (5) Plus Build/NPNS25",
                   "Trident/7.0", "XYZ-123", None])
    out = N.device_brand(s).tolist()
    assert out[:7] == ["samsung", "apple ios", "windows", "apple mac", "motorola", "windows", "other"]
    assert pd.isna(out[7])


def test_email_group():
    s = pd.Series(["gmail.com", "hotmail.com", "outlook.com", "ymail.com", "anonymous.com", "comcast.net", None])
    out = N.email_group(s).tolist()
    assert out[:6] == ["gmail", "microsoft", "microsoft", "yahoo", "anonymous", "other"]
    assert pd.isna(out[6])


def test_amount_decimals():
    s = pd.Series([10.0, 10.5, 10.25, 10.125, 49.0, 117.008, 0.251])
    assert N.amount_decimals(s).tolist() == [0, 1, 2, 3, 0, 3, 3]
