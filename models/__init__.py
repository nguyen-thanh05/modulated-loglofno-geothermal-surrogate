"""Model package exports.

The heavy model modules are imported lazily so lightweight utilities can import
submodules without initializing optional dependencies.
"""

__all__ = ["ModulatedLOGLO_FNO", "FNOWrapper", "UNOWrapper", "UNet3D"]


def __getattr__(name):
    if name == "ModulatedLOGLO_FNO":
        from .loglo_fno import ModulatedLOGLO_FNO

        return ModulatedLOGLO_FNO
    if name == "FNOWrapper":
        from .fno_wrapper import FNOWrapper

        return FNOWrapper
    if name == "UNOWrapper":
        from .uno_wrapper import UNOWrapper

        return UNOWrapper
    if name == "UNet3D":
        from .unet3d import UNet3D

        return UNet3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
