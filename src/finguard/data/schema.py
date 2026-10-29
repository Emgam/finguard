"""Expected schema of the IEEE-CIS files and the column groups used throughout the project.

Column meanings are NOT invented here. The competition host disclosed only group-level
descriptions (e.g. "C1-C14: counting", "D1-D15: timedelta", "Vxxx: Vesta engineered
features"); everything else is treated as undisclosed.
"""
from __future__ import annotations

TXN_BASE = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
            "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
            "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
V_COLS = [f"V{i}" for i in range(1, 340)]
TRAIN_TXN_COLS = TXN_BASE + C_COLS + D_COLS + M_COLS + V_COLS            # 394 columns
TEST_TXN_COLS = [c for c in TRAIN_TXN_COLS if c != "isFraud"]          # 393 columns

ID_COLS = [f"id_{i:02d}" for i in range(1, 39)]
IDENTITY_COLS = ["TransactionID"] + ID_COLS + ["DeviceType", "DeviceInfo"]  # 41 columns

# Text (categorical) columns in the joined table.
TEXT_COLS = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] + M_COLS
             + ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31",
                "id_33", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"])

# Numeric columns that are really category codes (arithmetic on them is meaningless).
NUMERIC_CODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

# Never use as model features (see the leakage section of the data-quality report).
EXCLUDE_FROM_FEATURES = ["TransactionID", "isFraud", "TransactionDT"]

# Disclosure level of each group, used for the interpretability column of the inventory.
GROUP_DISCLOSURE = {
    "TransactionAmt": "readable",
    "TransactionDT": "readable (relative seconds)",
    "ProductCD": "code (letters undisclosed)",
    "card1-6": "card4/card6 readable; card1/2/3/5 codes",
    "addr1-2": "code (no coordinates or country names)",
    "dist1-2": "undisclosed distance",
    "email": "readable",
    "DeviceType/Info": "readable",
    "id_01-38": "mostly masked; id_30/id_31/id_33 readable",
    "C1-14": "masked (host: 'counting')",
    "D1-15": "masked (host: 'timedelta')",
    "M1-9": "masked (host: 'match')",
    "V1-339": "masked (host: 'Vesta engineered')",
}


def feature_group(col: str) -> str:
    """Map a column to its documented feature group."""
    if col in ("TransactionDT", "TransactionAmt", "ProductCD"):
        return col
    if col.startswith("card"):
        return "card1-6"
    if col.startswith("addr"):
        return "addr1-2"
    if col.startswith("dist"):
        return "dist1-2"
    if "emaildomain" in col:
        return "email"
    if col.startswith("Device"):
        return "DeviceType/Info"
    if col.startswith("id_"):
        return "id_01-38"
    if col[0] in "CDMV" and col[1:].isdigit():
        return {"C": "C1-14", "D": "D1-15", "M": "M1-9", "V": "V1-339"}[col[0]]
    return "other"
