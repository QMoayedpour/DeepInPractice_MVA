import torch
import torch_geometric

def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    
    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (for a 3x3 kernel with padding=1, stride=1)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    C, H, W = image.shape
    # Node features: flatten image pixels (each node gets the C-dimensional feature vector of a pixel)
    x = image.permute(1, 2, 0).reshape(H * W, C)

    # Construct edges and edge attributes for a 3x3 neighborhood around each pixel
    source_indices = []
    target_indices = []
    edge_attrs = []
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),  (0, 0),  (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
    offset_to_index = {offset: idx for idx, offset in enumerate(neighbor_offsets)}
    for i in range(H * W):
        r = i // W
        c = i % W
        for (dr, dc) in neighbor_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                j = nr * W + nc
                source_indices.append(j)
                target_indices.append(i)
                edge_attrs.append(offset_to_index[(dr, dc)])
            # If neighbor is outside bounds, skip (zero-padding effect)

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (for a 3x3 kernel with padding=1, stride=1)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    N, C = data.shape
    assert N == height * width, "Mismatch between number of nodes and image dimensions."
    image = data.view(height, width, C).permute(2, 0, 1).contiguous()
    return image


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """
    def __init__(self, conv2d: torch.nn.Conv2d):
        # Initialize with sum aggregation
        super(Conv2dMessagePassing, self).__init__(aggr='add')
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        # Prepare conv weights for message passing (reshape by offsets)
        weight = conv2d.weight.data.clone()
        k_h, k_w = conv2d.kernel_size
        weight = weight.reshape(self.out_channels, self.in_channels, k_h * k_w)
        weight = weight.permute(2, 0, 1).contiguous()
        self.register_buffer('weight_by_offset', weight)
        if conv2d.bias is not None:
            self.register_buffer('bias', conv2d.bias.data.clone().view(1, -1))
        else:
            self.bias = None

    def forward(self, data):
        self.edge_index = data.edge_index
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph (source u to target v),
        the message through the edge e is calculated as the convolution weight corresponding to that edge's relative position times the source node's features.
        (The message is φ(u, v, e) in the formalism.)
        
        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the source node for each edge (shape: E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (shape: E, each an offset index).
        
        Returns:
        --------
        torch.Tensor
            The message for each edge (shape: E x out_channels).
        """
        w = self.weight_by_offset[edge_attr]
        message = (w * x_j.unsqueeze(1)).sum(dim=2)
        return message
