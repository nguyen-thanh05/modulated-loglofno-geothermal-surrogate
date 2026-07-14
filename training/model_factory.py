from models.unet3d import UNet3D
from models.fno_wrapper import FNOWrapper
from models.ufno import UFNO3D
from models.uno_wrapper import UNOWrapper
from models.loglo_fno import ModulatedLOGLO_FNO, VanillaLOGLO_FNO


def create_model(model_cfg, model_type):
    if model_type == 'unet3d':
        return UNet3D(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_channels=model_cfg['hidden_channels'],
            depth=model_cfg.get('depth', 3),
            channel_multipliers=model_cfg.get('channel_multipliers', None),
            n_blocks=model_cfg.get('n_blocks', 2),
            norm=model_cfg.get('norm', True),
            activation=model_cfg.get('activation', 'gelu'),
            mid_attn=model_cfg.get('mid_attn', False),
            is_attn=model_cfg.get('is_attn'),
            use1x1=model_cfg.get('use1x1', True),
        )
    elif model_type == 'fno':
        return FNOWrapper(
            n_modes=model_cfg['n_modes'],
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            n_layers=model_cfg['n_layers'],
            hidden_channels=model_cfg['hidden_channels'],
        )
    elif model_type == 'ufno':
        return UFNO3D(
            n_modes=model_cfg['n_modes'],
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            n_layers=model_cfg['n_layers'],
            hidden_channels=model_cfg['hidden_channels'],
            n_unet_layers=model_cfg['n_unet_layers'],
            lifting_channels=model_cfg.get('lifting_channels', 128),
            projection_channels=model_cfg.get('projection_channels', 128),
            unet_dropout=model_cfg.get('unet_dropout', 0.0),
        )
    elif model_type == 'uno':
        return UNOWrapper(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_channels=model_cfg['hidden_channels'],
            n_layers=model_cfg['n_layers'],
            uno_out_channels=model_cfg['uno_out_channels'],
            uno_n_modes=model_cfg['uno_n_modes'],
            uno_scalings=model_cfg['uno_scalings'],
            horizontal_skips_map=model_cfg.get('horizontal_skips_map'),
            channel_mlp_skip=model_cfg.get('channel_mlp_skip', 'linear'),
        )
    elif model_type == 'modulated_loglo':
        return ModulatedLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
            action_channels=model_cfg['action_channels'],
        )
    elif model_type == 'vanilla_loglo':
        return VanillaLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_out_channels(model_cfg):
    return model_cfg.get('out_channels', model_cfg.get('out_dim', 4))


def get_hidden_dim(model_cfg):
    return model_cfg.get('hidden_channels', model_cfg.get('hidden_dim', 64))
