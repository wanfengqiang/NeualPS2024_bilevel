import torch
import torch.nn as nn
import torch.nn.functional as F

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.t_in = nn.Linear(768,512)
        self.v_in = nn.Linear(1000,512)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        x = self.t_in(x)
        y = self.v_in(y)
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output
    

class ConcatThree(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatThree, self).__init__()
        self.fc_out = nn.Linear(input_dim*3, output_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        output = self.fc_out(output)
        return x, y, z, output

class Constractive(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(Constractive, self).__init__()

        self.t_in = nn.Linear(768,512)
        self.v_in = nn.Linear(1000,512)
        self.fc_out = nn.Linear(input_dim, output_dim)

        self.itm_head = nn.Linear(input_dim,2)
        

    # def forward(self, x, y):

    #     x = self.t_in(x)
    #     y = self.v_in(y)


    #     index_matrix = torch.mm(x, y.T)
    #     index_matrix.fill_diagonal_(float('-inf'))
    #     neg_x_index = torch.argmax(index_matrix, axis=1)
    #     neg_y_index = torch.argmax(index_matrix, axis=0)

    #     neg_x = x[neg_x_index]
    #     neg_y = y[neg_y_index]
    #     bs = x.size(0)
    #     pos = torch.cat((x, y), dim=1)
    #     neg_i2t = torch.cat((x, neg_y), dim=1)
    #     neg_t2i = torch.cat((neg_x, y), dim=1)

    #     itm_feat = torch.cat((pos, neg_i2t, neg_t2i), dim=0)
    #     itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],dim=0).cuda()
    #     loss_itm = F.cross_entropy(itm_feat,itm_labels)
    #     output = torch.cat((x, y), dim=1)
    #     output = self.fc_out(output)
    #     return x, y, output, loss_itm


    def forward(self, x, y):
        x = self.t_in(x)  # Encode text
        y = self.v_in(y)  # Encode images

        # Normalize features to unit length
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        # Compute similarity logits
        logits_per_text = x @ y.T
        logits_per_image = y @ x.T

        # Ground-truth for contrastive loss
        batch_size = x.size(0)
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=x.device)

        # Cross entropy loss
        loss_text = F.cross_entropy(logits_per_text, ground_truth)
        loss_image = F.cross_entropy(logits_per_image, ground_truth)

        # Final loss
        loss = (loss_text + loss_image) / 2.0

        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output, loss





class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output

