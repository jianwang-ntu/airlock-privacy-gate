"""Identifier types Airlock detects, and the BIO tag set derived from them.

The type list is the label vocabulary of the training corpus
(gretelai/synthetic_pii_finance_multilingual, Apache-2.0) and is read from the
corpus rather than invented here; `scripts/check_label_vocab.py` re-derives it
from the parquet files and fails if this list drifts.
"""

PII_TYPES = [
    "account_pin",
    "api_key",
    "bank_routing_number",
    "bban",
    "company",
    "credit_card_number",
    "credit_card_security_code",
    "customer_id",
    "date",
    "date_of_birth",
    "date_time",
    "driver_license_number",
    "email",
    "employee_id",
    "iban",
    "ipv4",
    "ipv6",
    "first_name",
    "last_name",
    "local_latlng",
    "name",
    "passport_number",
    "password",
    "phone_number",
    "ssn",
    "street_address",
    "swift_bic_code",
    "time",
    "user_name",
]

# Types whose disclosure identifies or authenticates a person. `company`,
# `date`, `time` and `date_time` are annotated by the corpus but are not
# direct identifiers, so they are detected and reported separately and do not
# count as leaks. Keeping that distinction explicit stops the headline leak
# number from being flattered by easy calendar dates.
NON_IDENTIFYING = {"company", "date", "time", "date_time"}
IDENTIFYING = [t for t in PII_TYPES if t not in NON_IDENTIFYING]

LABELS = ["O"] + [f"{p}-{t}" for t in PII_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def type_of(label: str) -> str:
    return "" if label == "O" else label[2:]
