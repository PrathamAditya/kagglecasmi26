import pytest

from casmi.chem import NEUTRAL_OFFSET, metric_ik14, neutral_mass

GLUCOSE = 180.063388  # C6H12O6
IONS = {  # independently computed ion m/z for glucose
    "[M+H]+": 181.070664, "[M+Na]+": 203.052609, "[M-H]-": 179.056112, "[M+Cl]-": 215.032790,
    "[M+CH2O2-H]-": 225.061592, "[M-H2O+H]+": 163.060099, "[M+NH4]+": 198.097213, "[M+K]+": 219.026546,
    "[M-2H2O+H]+": 145.049534, "[M-H2O-H]-": 161.045547,
}


def test_all_ten_adducts_covered():
    assert set(IONS) == set(NEUTRAL_OFFSET)


@pytest.mark.parametrize("adduct,mz", IONS.items())
def test_neutral_mass(adduct, mz):
    assert abs(neutral_mass(mz, adduct) - GLUCOSE) < 2e-5


def test_metric_ignores_stereo():
    a = metric_ik14("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
    b = metric_ik14("OCC1OC(O)C(O)C(O)C1O")
    assert a == b == "WQZGKKKJIJFFOK"
