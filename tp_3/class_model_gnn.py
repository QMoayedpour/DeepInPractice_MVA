import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graphnn


# Define model ( in your class_model_gnn.py)

class StudentModel(nn.Module):
    def __init__(self, i_size=50, h_size=256//2, o_size=121, n_heads=4):
        '''
        From the articles provided, we use the "GATConv" instead of the GCN Conv
        We add some linear layers (fc_1, fc_2, o_2) to "add" information to the model
        If we remove them and keep only the GATConv, we get better results than the basic model
        but we do not atteign 93% (note: maybe we can atteign it with some hyperparameters tuning
        but by just adding those linear layers, we get good results)
        PS: on 100 epochs, with the model only with the gatconv, we get 80% f1 score with 4 heads and
        h_size=128
        '''
        super().__init__()
        # graph 
        self.attn_1 = graphnn.GATConv(i_size, h_size, heads = n_heads)
        self.attn_2 = graphnn.GATConv(n_heads*h_size, h_size, heads = n_heads)
        self.o_1 = graphnn.GATConv(n_heads*h_size, o_size, heads = n_heads, concat=False)
        # o stands for "out"

        self.fc_1 = nn.Linear(i_size, n_heads*h_size) # fc stands for function
        # but usually i denote the linear layers with fc...
        self.fc_2 = nn.Linear(n_heads*h_size, n_heads*h_size)
        self.o_2 = nn.Linear(n_heads*h_size, o_size)

        self.relu = nn.LeakyReLU() # Activation function, in practice (in our case!!!) doesn't really
        # change if we select leakyrelu, elu etc...
        
    def forward(self, x, edge_index):
        x = self.attn_1(x, edge_index) + self.fc_1(x) # the idea here is to add a residual connection
        # but x is not the same shape as attn_1(x) so we use a linear layer to have the same shape
        x = self.relu(x)
        x = self.attn_2(x, edge_index) + self.fc_2(x)
        x = self.relu(x)
        x = self.o_1(x, edge_index) + self.o_2(x)

        return x