import pytest
from src.algorithms.qr_parse import qr_parse
from src.algorithms.types import TaskPlan, QrFormatError

def test_parse_valid():
    plan = qr_parse("452+321+254+312")
    assert plan == TaskPlan((4,5,2),(3,2,1),(2,5,4),(3,1,2))

@pytest.mark.parametrize("bad", ["", "45+321+254+312", "452+321+254", "452+321+254+317",
                                  "abc+321+254+312", "452+321+254+310"])
def test_parse_invalid(bad):
    with pytest.raises(QrFormatError):
        qr_parse(bad)
