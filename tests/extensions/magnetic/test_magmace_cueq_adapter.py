"""Smoke tests for the standalone CuEq magnetic MACE adapter (``magmace_cueq``).

These tests do **not** require ``cuequivariance`` to be installed: on CPU / without
the CuEq kernels the adapter transparently falls back to the pure-e3nn path, which
is exactly what we want to verify here. When CuEq *is* available the same tests
exercise the accelerated path.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.modules.extensions import MagneticScaleShiftMACE
from mace.tools.torch_tools import default_dtype

import magmace_cueq
from magmace_cueq import (
    CUEQ_AVAILABLE,
    MagneticScaleShiftMACE_CuEq,
    build_tiny,
    extract_config,
    make_cueq_config,
    to_cueq_model,
)


def _make_graph() -> dict:
    """A minimal 2-atom Fe dimer graph in the format ``forward`` expects."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    node_attrs = torch.eye(1, dtype=torch.float32)  # one element, one-hot
    node_attrs = torch.cat([node_attrs, node_attrs])  # (2, 1)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    shifts = torch.zeros(2, 3)
    magmom = torch.tensor([[0.0, 0.0, 2.2], [0.0, 0.0, 2.2]])
    data = {
        "positions": positions,
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "shifts": shifts,
        "magmom": magmom,
        "batch": torch.tensor([0, 0]),
        "ptr": torch.tensor([0, 2]),
        "cell": torch.eye(3) * 6.0,
        "unit_shifts": torch.zeros(2, 3),
    }
    return data


@pytest.fixture(name="cueq_model")
def fixture_cueq_model():
    with default_dtype(torch.float32):
        return build_tiny(cueq_config=make_cueq_config(enabled=CUEQ_AVAILABLE))


def test_build_and_forward(cueq_model):
    model = cueq_model
    assert isinstance(model, MagneticScaleShiftMACE_CuEq)
    assert isinstance(model, MagneticScaleShiftMACE)

    data = _make_graph()
    out = model(data, compute_force=True, compute_magforces=True)

    assert "energy" in out
    assert "forces" in out
    assert "magforces" in out
    assert torch.isfinite(out["energy"]).all()
    assert out["energy"].shape[0] == 1  # single graph


def test_cueq_active_flag_matches_availability(cueq_model):
    # The flag must reflect whether CuEq kernels were actually used.
    assert cueq_model.cueq_active is CUEQ_AVAILABLE


def test_extract_config_roundtrip(cueq_model):
    config = extract_config(cueq_model)
    # Core reconstruction keys must be present.
    for key in (
        "r_max",
        "num_bessel",
        "max_ell",
        "hidden_irreps",
        "MLP_irreps",
        "m_max",
        "max_m_ell",
        "num_mag_radial_basis",
        "atomic_energies",
        "atomic_inter_scale",
        "atomic_inter_shift",
    ):
        assert key in config, f"missing config key: {key}"


def test_to_cueq_model_lifts_weights(cueq_model):
    source = cueq_model
    lifted = to_cueq_model(source, layout="mul_ir", device="cpu")

    assert isinstance(lifted, MagneticScaleShiftMACE_CuEq)

    # A representative of the directly-transferable weights should match.
    src_sd = source.state_dict()
    tgt_sd = lifted.state_dict()
    common = set(src_sd) & set(tgt_sd)
    matched = sum(
        1
        for k in common
        if src_sd[k].shape == tgt_sd[k].shape
        and torch.allclose(src_sd[k], tgt_sd[k], atol=1e-5)
    )
    assert matched > 0, "no shared weights were transferred"

    # And the lifted model still produces finite energies.
    data = _make_graph()
    out = lifted(data, compute_force=False, compute_magforces=False)
    assert torch.isfinite(out["energy"]).all()


def test_from_plain_e3nn_model():
    """Lifting a plain (non-CuEq) e3nn magnetic MACE into the adapter works."""
    from mace.modules import interaction_classes

    with default_dtype(torch.float32):
        e3nn_model = MagneticScaleShiftMACE(
            r_max=3.5,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=2,
            interaction_cls=interaction_classes[
                "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
            ],
            interaction_cls_first=interaction_classes[
                "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
            ],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=1.0,
            atomic_numbers=[26],
            correlation=[1],
            gate=torch.nn.functional.silu,
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
            m_max=[3.0],
            num_mag_radial_basis=8,
            num_mag_radial_basis_one_body=10,
            max_m_ell=1,
            use_magmom_one_body=False,
        )

    lifted = to_cueq_model(e3nn_model, layout="mul_ir", device="cpu")
    assert isinstance(lifted, MagneticScaleShiftMACE_CuEq)

    data = _make_graph()
    out = lifted(data, compute_force=False, compute_magforces=False)
    assert torch.isfinite(out["energy"]).all()
