import torch

class DownsampleHook():
    def __init__(self, module, downsample=False, num_kept_neurons=0):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.downsample = downsample
        self.num_kept_neurons = num_kept_neurons
        self.indices = None
        self.indices_chosen = False


    def hook_fn(self, module, input, output):
        if self.downsample and not self.indices_chosen:
            # If this is the first time forward has been called
            num_neurons = torch.prod(torch.tensor(output.shape)[1:]).item()
            assert num_neurons >= self.num_kept_neurons
            # Randomly choose which neurons you want to keep
            self.indices = torch.randperm(num_neurons)[:self.num_kept_neurons]
            self.indices_chosen = True

        if self.downsample:
            self.output = output.flatten(start_dim=1)[:,self.indices]
        else:
            self.output = output.flatten(start_dim=1)

    def close(self):
        self.hook.remove()
