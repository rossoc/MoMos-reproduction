import torch.nn as nn


class MLP(nn.Module):
    """Simple feed-forward classifier over flattened image pixels."""

    def __init__(self, input_dim, num_classes):
        """Create an MLP classifier for flattened inputs.

        Args:
            input_dim: Flattened input dimension.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        """Run a forward pass after flattening each sample.

        Args:
            x: Input tensor ``(batch, channels, height, width)``.

        Returns:
            Logits tensor ``(batch, num_classes)``.
        """
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x


# return MLP(int(in_channels) * int(img_size) * int(img_size), num_classes)
