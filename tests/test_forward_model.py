"""Offline tests of the De Zan forward model."""
import numpy as np
import pytest

from insar4sm.dielectic_models import hallikainen_1985_calc
from insar4sm.forward_modelling_funcs import (vertical_wavenumber_calc, InSAR_coherence_model,
                                              Single_phase_closure_model)

C = 299792458.0


@pytest.mark.parametrize('freq_GHz', [5.405, 1.257])
def test_wavenumber_has_physical_magnitude(freq_GHz):
    """At normal incidence k = (omega/c) * sqrt(eps_r), with an attenuating (negative) imaginary part."""
    mv, clay, sand = 15.0, 11.0, 79.0
    k = vertical_wavenumber_calc(mv, None, freq_GHz, clay, sand)
    eps_r = hallikainen_1985_calc(clay, sand, mv, freq_GHz)
    expected = 2 * np.pi * freq_GHz * 1e9 / C * np.sqrt(eps_r)
    assert abs(k) == pytest.approx(abs(expected), rel=1e-3)
    assert np.imag(k) <= 0


@pytest.mark.parametrize('freq_GHz', [5.405, 1.257])
def test_coherence_is_one_for_equal_moisture_and_drops_with_change(freq_GHz):
    same = abs(InSAR_coherence_model(10, 10, 35.0, freq_GHz, 11, 79))
    small = abs(InSAR_coherence_model(10, 15, 35.0, freq_GHz, 11, 79))
    large = abs(InSAR_coherence_model(10, 35, 35.0, freq_GHz, 11, 79))
    assert same == pytest.approx(1.0, abs=1e-6)
    assert 1.0 > small > large > 0


def test_phase_closure_vanishes_when_two_moistures_are_equal():
    assert Single_phase_closure_model(10, 10, 30, 35.0, 5.405, 11, 79) == pytest.approx(0.0, abs=1e-9)
    assert abs(Single_phase_closure_model(5, 15, 30, 35.0, 5.405, 11, 79)) > 1e-3
