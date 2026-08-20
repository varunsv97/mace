"""Standalone cuEquivariance (CuEq) magnetic MACE adapter.

This package implements a **separate** CuEq-accelerated variant of the magnetic
MACE model *without editing any file in the ``mace`` source tree*. It reuses the
existing, already CuEq-aware building blocks (the wrapper ops in
``mace.modules.wrapper_ops`` and the magnetic interaction / product-basis blocks
in ``mace.modules.blocks``) but assembles them through its own model class so that:

* every equivariant primitive (linear maps, channelwise tensor products, fully
  connected tensor products, symmetric contractions) is routed to its
  ``cuequivariance_torch`` (``cuet.*``) implementation when CuEq is available and
  enabled, and transparently falls back to the pure-``e3nn`` path otherwise;
* an existing e3nn :class:`mace.modules.MagneticScaleShiftMACE` checkpoint can be
  lifted into this CuEq model via :func:`magmace_cueq.convert.to_cueq_model`.

The public surface is intentionally small:

.. code-block:: python

    from magmace_cueq import MagneticScaleShiftMACE_CuEq, to_cueq_model

    # build fresh with CuEq acceleration
    model = MagneticScaleShiftMACE_CuEq(**config, cueq_config=cueq_cfg)

    # or lift an existing e3nn magnetic MACE checkpoint
    cueq_model = to_cueq_model(e3nn_model, layout="mul_ir")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from e3nn import o3

from mace.modules import (
    MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
    MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
    NonLinearReadoutBlock,
)
from mace.modules.extensions import MagneticScaleShiftMACE
from mace.modules.wrapper_ops import CuEquivarianceConfig

try:  # pragma: no cover - depends on environment
    import cuequivariance  # noqa: F401

    CUEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    CUEQ_AVAILABLE = False


def make_cueq_config(
    enabled: bool = True,
    layout: str = "mul_ir",
    group: str = "O3_e3nn",
    optimize_all: bool = True,
    conv_fusion: bool = False,
) -> CuEquivarianceConfig:
    """Build a :class:`~mace.modules.wrapper_ops.CuEquivarianceConfig`.

    Parameters
    ----------
    enabled:
        Whether to route primitives to ``cuet.*``. Automatically downgraded to
        ``False`` when ``cuequivariance`` is not importable (see
        :data:`CUEQ_AVAILABLE`).
    layout:
        Irreps layout, either ``"mul_ir"`` or ``"ir_mul"``.
    group:
        Group name understood by the wrapper ops (``"O3"``, ``"SO3"`` or
        ``"O3_e3nn"``).
    optimize_all:
        Enable all CuEq optimizations (linear, channelwise TP, fctp, symmetric).
    conv_fusion:
        Fuse the scatter-sum into the convolution kernel (GPU only).
    """
    return CuEquivarianceConfig(
        enabled=enabled,
        layout=layout,
        group=group,
        optimize_all=optimize_all,
        conv_fusion=conv_fusion,
    )


class MagneticScaleShiftMACE_CuEq(MagneticScaleShiftMACE):
    """CuEq-accelerated magnetic MACE, assembled without touching ``mace`` source.

    This subclass does **not** override any forward logic — it simply guarantees
    that a valid :class:`CuEquivarianceConfig` is threaded through to every
    sub-module at construction time. Because the underlying blocks already branch
    on ``cueq_config.enabled`` inside :mod:`mace.modules.wrapper_ops`, passing a
    config here is sufficient to switch the whole network onto the CuEq kernels.

    All constructor arguments are identical to
    :class:`mace.modules.extensions.MagneticScaleShiftMACE`; see that class for
    the full signature. The only behavioural difference is the handling of
    ``cueq_config``:

    * if ``cueq_config`` is a dict, it is converted to a
      :class:`CuEquivarianceConfig` (with ``enabled`` defaulting to True);
    * if ``cueq_config`` is omitted, a default enabled config is created whenever
      :data:`CUEQ_AVAILABLE` is true, otherwise ``None`` (pure e3nn).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        cueq_config = kwargs.pop("cueq_config", None)
        if isinstance(cueq_config, dict):
            cueq_config = CuEquivarianceConfig(**cueq_config)
        elif cueq_config is None:
            cueq_config = make_cueq_config(enabled=CUEQ_AVAILABLE)
        kwargs["cueq_config"] = cueq_config
        super().__init__(*args, **kwargs)

    @property
    def cueq_active(self) -> bool:
        """True when the network was built on top of CuEq kernels."""
        cfg = getattr(self, "cueq_config", None)
        return bool(cfg is not None and getattr(cfg, "enabled", False))


def build_tiny(
    cueq_config: Optional[CuEquivarianceConfig] = None,
    num_interactions: int = 1,
    correlation: int = 1,
) -> MagneticScaleShiftMACE_CuEq:
    """Build a tiny magnetic MACE model for tests / quick experiments.

    Mirrors the minimal configuration used in ``tests/extensions/magnetic`` so the
    adapter can be exercised without a real dataset.
    """
    import numpy as np

    from mace.modules import interaction_classes

    return MagneticScaleShiftMACE_CuEq(
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
        num_interactions=num_interactions,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=1.0,
        atomic_numbers=[26],
        correlation=[correlation] * num_interactions,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=10,
        max_m_ell=1,
        use_magmom_one_body=False,
        cueq_config=cueq_config,
    )


def extract_config(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract a reconstruction config from an existing magnetic MACE model.

    Thin wrapper around :func:`mace.tools.scripts_utils.extract_config_mace_model`
    that additionally records whether the source model was CuEq-enabled, so the
    conversion helpers below can decide what to do.
    """
    from mace.tools.scripts_utils import extract_config_mace_model

    # The upstream extractor dispatches on the *exact* class name, which would
    # reject this adapter's subclass (``MagneticScaleShiftMACE_CuEq``). Rebind the
    # class name temporarily so the extractor treats it as a plain
    # ``MagneticScaleShiftMACE``; no source file is modified.
    original_name = model.__class__.__name__
    try:
        if not isinstance(model, MagneticScaleShiftMACE):
            raise ValueError(
                f"Model of type {original_name} is not a MagneticScaleShiftMACE."
            )
        model.__class__.__name__ = "MagneticScaleShiftMACE"
        config = extract_config_mace_model(model)
    finally:
        model.__class__.__name__ = original_name

    if "error" in config:
        raise ValueError(f"Cannot extract config: {config['error']}")
    cfg = getattr(model, "cueq_config", None)
    config["_source_cueq_enabled"] = bool(
        cfg is not None and getattr(cfg, "enabled", False)
    )
    return config


def _transfer_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy weights from ``source`` into ``target`` where shapes line up.

    Symmetric-contraction weight layouts differ between the reduced-CG CuEq form
    and the plain e3nn form, so those keys are transferred using the same
    projection routine the official converter uses
    (:func:`mace.cli.convert_e3nn_cueq.transfer_symmetric_contractions`) when the
    source is a plain e3nn model and the target is CuEq. Everything else is copied
    directly when the shapes match.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()

    correlation = int(target.products[0].symmetric_contractions.contraction_degree) \
        if hasattr(target.products[0].symmetric_contractions, "contraction_degree") \
        else len(target.products[0].symmetric_contractions.contractions[0].weights) + 1
    num_layers = len(target.products)
    use_reduced_cg = getattr(target, "use_reduced_cg", True)

    try:
        from mace.cli.convert_e3nn_cueq import transfer_symmetric_contractions

        transfer_symmetric_contractions(
            source_dict,
            target_dict,
            num_product_irreps=len(
                o3.Irreps(str(target.products[0].linear.irreps_out)).slices()
            )
            - 1,
            products=target.products,
            correlation=correlation,
            num_layers=num_layers,
            use_reduced_cg=use_reduced_cg,
        )
    except Exception:  # pragma: no cover - best effort
        pass

    transferred = {k for k in target_dict if "symmetric_contraction" in k}
    for key in set(source_dict.keys()) & set(target_dict.keys()):
        if key in transferred:
            continue
        if source_dict[key].shape == target_dict[key].shape:
            target_dict[key] = source_dict[key]

    target.load_state_dict(target_dict, strict=False)
    for i in range(len(target.interactions)):
        target.interactions[i].set_avg_num_neighbors(
            source.interactions[i].avg_num_neighbors
        )


def to_cueq_model(
    model: torch.nn.Module,
    layout: str = "mul_ir",
    device: str = "cpu",
    conv_fusion: bool = False,
) -> MagneticScaleShiftMACE_CuEq:
    """Lift an existing (e3nn or CuEq) magnetic MACE model into the CuEq adapter.

    Parameters
    ----------
    model:
        A :class:`mace.modules.extensions.MagneticScaleShiftMACE` (or the CuEq
        subclass). May be loaded from a checkpoint via ``torch.load``.
    layout:
        Target irreps layout (``"mul_ir"`` or ``"ir_mul"``).
    device:
        Device to place the resulting model on.
    conv_fusion:
        Enable convolution/scatter fusion (requires CUDA).

    Returns
    -------
    MagneticScaleShiftMACE_CuEq
        A new model whose equivariant primitives are backed by CuEq kernels when
        available, with weights transferred from ``model``.
    """
    config = extract_config(model)
    config.pop("_source_cueq_enabled", None)
    # The extractor records the source model's own config; drop it so we can pass
    # a fresh, explicitly-constructed CuEq config below without a kwarg collision.
    config.pop("cueq_config", None)

    cueq_config = make_cueq_config(
        enabled=CUEQ_AVAILABLE,
        layout=layout,
        conv_fusion=bool(conv_fusion and torch.device(device).type == "cuda"),
    )

    target = MagneticScaleShiftMACE_CuEq(**config, cueq_config=cueq_config).to(device)
    _transfer_state(model, target)
    return target


__all__ = [
    "CUEQ_AVAILABLE",
    "MagneticScaleShiftMACE_CuEq",
    "make_cueq_config",
    "extract_config",
    "to_cueq_model",
    "MagneticScaleShiftMACE",
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
    "NonLinearReadoutBlock",
    "CuEquivarianceConfig",
]
