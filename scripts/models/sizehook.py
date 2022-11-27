import torch

class SizeHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output.flatten(start_dim=1).shape[1]

    def close(self):
        self.hook.remove()
