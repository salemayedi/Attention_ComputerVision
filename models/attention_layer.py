import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, att_type):
        super(Attention, self).__init__()
        if att_type=='additive':
            self.attention_layer = AdditiveAttention(encoder_dim)
        elif att_type=='dotprod':
            self.attention_layer = DotProdAttention(encoder_dim)
        else:
            raise ValueError('Attention mechanism not defined: '+str(att_type))

    def forward(self, encoder_output, hidden_state, sum_context=False):
        context, alpha = self.attention_layer(encoder_output, hidden_state)
        if sum_context:
            context = context.sum(1)
        return context, alpha

# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(AdditiveAttention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, encoder_output, hidden_state):
        # encoder_output ------ torch.Size([24, 49, 2048])
        # hidden_state ------ torch.Size([24, 512])
        U_h = self.U(hidden_state).unsqueeze(1)
        # Uh ------ torch.Size([24, 1, 512])
        W_s = self.W(encoder_output)
        # Ws (s=encoder_output) ------ torch.Size([24, 49, 512])
        att = self.tanh(W_s + U_h)
        # a = tanh(Ws + Uh) ------ torch.Size([24, 49, 512])
        e = self.v(att).squeeze(2)
        # e = V^tanh(Ws + Uh) ------ torch.Size([24, 49])
        alpha = self.softmax(e)
        # alpha ------ torch.Size([24, 49])
        context = (encoder_output * alpha.unsqueeze(2))
        # context ------ torch.Size([24, 49, 2048])
        return context, alpha

# Dot-Product Based or General Multiplicative Attention
class DotProdAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(DotProdAttention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.softmax = nn.Softmax(1)
        #raise NotImplementedError("TODO: Implement attention layer")

    def forward(self, encoder_output, hidden_state):
        # encoder_output ------ torch.Size([2, 256, 512])
        # hidden_state ------ torch.Size([2, 512])
        hidden_state = self.U(hidden_state)

        hidden_state = hidden_state.unsqueeze(1) # torch.Size([2, 1, 512])
        hidden_state = hidden_state.permute(0,2,1) # torch.Size([2, 512, 1])
        att = torch.matmul(encoder_output, hidden_state)  # torch.Size([2, 256, 512]) * torch.Size([2, 512, 1])  
                                        # = torch.Size([2, 256, 1])
                                        # att are the scores
        e = att.squeeze(2) #  torch.Size([2, 256]) 
        alpha = self.softmax(e) # convert att to probabilities
        # alpha ------ torch.Size([2, 256])
        context = (encoder_output * alpha.unsqueeze(2)) # attention
        # context ------ torch.Size([2, 256, 512])
        
        return context, alpha

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Attention(512, 'dotprod').to(device)
    model.eval()
    print(model)
    encoder_output = torch.randn(1, 256, 512).to(device)
    v_embedding = torch.randn(1, 512).to(device)
    with torch.no_grad():
        output, alpha = model.forward(encoder_output, v_embedding)
    print(output.size())
    print(alpha.size())
