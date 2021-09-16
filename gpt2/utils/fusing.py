try:
    print('Trying to use APEX ...')
    from apex.optimizers import FusedAdam as Adam
    from apex.normalization import FusedLayerNorm as LayerNorm
    print('Using APEX.')
except ModuleNotFoundError:
    print('Exception: Mudule \'APEX\' not found')
    from torch.optim import AdamW as Adam
    from torch.nn import LayerNorm
    print('Using Pytorch LayerNorm.')