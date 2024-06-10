
class TeUniform:

    for embedding in self.backbone.te.values():
        nn.init.normal_(embedding.weight, 0, 1)