"""Offline tests of the ISMN reader (synthetic values, no ISMN data)."""
import numpy as np
from insar4sm.ismn import read_ismn


def test_stm_keeps_only_good_values(tmp_path):
    f = tmp_path / 'station.stm'
    f.write_text('NET NET Station 33.0 -115.0 100.0 0.05 0.05 Sensor\n'
                 '2026/01/01 01:00 0.2000 G M\n2026/01/01 00:00 0.1000 G M\n2026/01/01 02:00 0.9000 C03 M\n')
    sm = read_ismn(str(f))
    assert np.allclose(sm.values, [0.1, 0.2])
    assert sm.index.is_monotonic_increasing


def test_csv(tmp_path):
    f = tmp_path / 'station.csv'
    f.write_text('date_time,soil_moisture\n2026-01-01 00:00:00,0.1\n2026-01-01 01:00:00,\n')
    assert read_ismn(str(f)).tolist() == [0.1]
