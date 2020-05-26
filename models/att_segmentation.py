import random
import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
try:
    from deeplab_decoder import Decoder
    from attention_layer import Attention
except ModuleNotFoundError:
    from .deeplab_decoder import Decoder
    from .attention_layer import Attention


class AttSegmentator(nn.Module):

    def __init__(self, num_classes, encoder, att_type='additive', img_size=(512, 512)):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.low_feat = IntermediateLayerGetter(encoder, {"layer1": "layer1"}).to(self.device)
        self.encoder = IntermediateLayerGetter(encoder, {"layer4": "out"}).to(self.device)
        # For resnet18
        encoder_dim = 512
        low_level_dim = 64
        self.num_classes = num_classes

        self.class_encoder = nn.Linear(num_classes, 512)

        self.attention_enc = Attention(encoder_dim, att_type)

        self.decoder = Decoder(2, encoder_dim, img_size, low_level_dim=low_level_dim, rates=[1, 6, 12, 18])

    def forward(self, x, v_class, out_att=False):

        #raise NotImplementedError("TODO: Implement the attention-based segmentation network")
        # Write the forward pass of the model.
        # Base the model on the segmentation model and add the attention layer.
        # Be aware of the dimentions.
        # x_enc, attention = self.attention_enc(x_enc, class_vec)

        # if out_att:
        #     return segmentation, attention
        # return segmentation


        self.low_feat.eval()
        self.encoder.eval()

        with torch.no_grad():
            # This is possible since gradients are not being updated
            low_level_feat = self.low_feat(x)['layer1']
            enc_feat = self.encoder(x)['out'] # encoder output: value tensor [1, 512, 16, 16]
        

        x_enc= enc_feat.permute(0, 2, 3, 1).contiguous() # encoder output: value tensor [1, 16, 16, 512]
        x_enc = x_enc.view(x_enc.shape[0], -1, x_enc.shape[-1]) # x_enc.view(batch_size, -1, n_features) [1, 16*16, 512]
        class_vec = self.class_encoder(v_class) # Hidden states: this is query tensor  [1, 512]
        x_enc, attention = self.attention_enc(x_enc, class_vec) 
        # x_enc: torch.Size([1, 16*16, 512]) , attention: torch.Size([1, 16*16y])
        x_enc = x_enc.permute(0, 2, 1).contiguous().view(enc_feat.shape)
        # x_enc = torch.Size([1, 512, 16,16]) 
        segmentation = self.decoder(x_enc, low_level_feat)

        # if self.num_classes==1:
        #     segmentation = torch.sigmoid(segmentation)
        if out_att:
            return segmentation, attention
        return segmentation

if __name__ == "__main__":
    from torchvision.models.resnet import resnet18
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_model = resnet18(num_classes=4).to(device)
    model = AttSegmentator(10, pretrained_model, att_type='dotprod' ).to(device)
    model.eval()
    print(model)
    image = torch.randn(1, 3, 512, 512).to(device)
    v_class = torch.randn(1, 10).to(device)
    with torch.no_grad():
        output = model.forward(image, v_class)
    print(output.size())
