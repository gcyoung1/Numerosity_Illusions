import torch

class DownsampleHook():
    def __init__(self, module, indices):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.indices = indices

    def hook_fn(self, module, input, output):
        self.output = output.flatten(start_dim=1)[:,self.indices]

    def close(self):
        self.hook.remove()
