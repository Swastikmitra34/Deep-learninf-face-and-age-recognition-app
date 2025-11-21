class FinetunedResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.fc_age = nn.Linear(in_features, 1)
        self.fc_gender = nn.Linear(in_features, 2)

    def forward(self, x):
        x = self.base(x)
        age = self.fc_age(x).squeeze(1)
        gender = self.fc_gender(x)
        return age, gender
