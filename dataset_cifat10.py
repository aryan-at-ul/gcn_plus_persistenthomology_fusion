import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from torchvision import datasets, transforms
from torch_geometric.utils import grid
# from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, DataLoader, Data

class CIFAR10Graph(InMemoryDataset):
    def __init__(self, root, is_train=True, transform=None, pre_transform=None, download=True):
        self.is_train = is_train
        self.dataset = datasets.CIFAR10(root=root, train=is_train, transform=transforms.ToTensor(), download=download)
        super(CIFAR10Graph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0] if is_train else self.processed_paths[1])

    @property
    def processed_file_names(self):
        return ['cifar10_train_graph.pt', 'cifar10_test_graph.pt']

    def _check_processed_files(self):
        return all(os.path.isfile(os.path.join(self.processed_dir, f)) for f in self.processed_file_names)

    def process(self):
        edge_index, pos = grid(32, 32)

        data_list = []
        for x, y in self.dataset:
            y_tensor = torch.tensor(y, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x=x.view(-1, 3), edge_index=edge_index, pos=pos, y=y_tensor)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0] if self.is_train else self.processed_paths[1])

# Usage:
train_dataset = CIFAR10Graph('./data', is_train=True, download=True)
test_dataset = CIFAR10Graph('./data', is_train=False, download=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

print(train_dataset[0])






# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)  # for reproducibility
#         self.conv1 = GCNConv(3, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, 10)  # 10 classes in MNIST

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = global_mean_pool(x, batch)  # Pool node features to get one per graph
#         return F.log_softmax(x, dim=1)

# model = GCN(hidden_channels=16)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, DynamicEdgeConv
import numpy as np

# def aggregate_edge_attributes(edge_attr, edge_index, num_nodes):
#     # Create an empty tensor to store aggregated edge attributes for each node
#     aggregated_edge_attr = torch.zeros((num_nodes, edge_attr.size(1)), device=edge_attr.device)

#     # Use advanced indexing for efficient aggregation
#     source_nodes, dest_nodes = edge_index
#     aggregated_edge_attr.index_add_(0, dest_nodes, edge_attr)

#     # Count the number of edges connected to each node for averaging
#     edge_count = torch.zeros(num_nodes, device=edge_attr.device)
#     edge_count.index_add_(0, dest_nodes, torch.ones_like(dest_nodes, dtype=torch.float))

#     # Avoid division by zero for isolated nodes
#     edge_count[edge_count == 0] = 1

#     # Compute the mean of edge attributes for each node
#     aggregated_edge_attr = aggregated_edge_attr / edge_count.unsqueeze(1)

#     return aggregated_edge_attr

# # Rest of the classes remain the same



# class DynamicFilterLayer(MessagePassing):
#     def __init__(self, in_channels, out_channels, k):
#         super(DynamicFilterLayer, self).__init__(aggr='max')  # Specify aggregation method
#         self.k = k
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * in_channels, out_channels),
#             nn.ReLU(),
#             nn.Linear(out_channels, out_channels)
#         )

#     def forward(self, x, edge_index):
#         edge_attr = torch.cat((x[edge_index[0, :]], x[edge_index[1, :]]), dim=1)
#         edge_attr = self.mlp(edge_attr)
#         return self.propagate(edge_index, x=edge_attr)  # Return the result of propagation


# class DynamicFilterBasedFeatures(MessagePassing):
#     def __init__(self, in_channels, hidden_channels,k):
#         super(DynamicFilterBasedFeatures, self).__init__(aggr='add')
#         self.k = k
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * in_channels, 3),  # Adjusted to output size 3
#             nn.ReLU(),
#             nn.Linear(3, 3)
#         )
#         self.conv = DynamicEdgeConv(self.mlp, k)
#         self.project = nn.Linear(in_channels + 3, 3)  # Projection layer adjusted

#     def forward(self, x, edge_index, batch=None):
#         edge_attr = self.conv(x)

#         # Aggregate edge attributes per node
#         aggregated_edge_attr = aggregate_edge_attributes(edge_attr, edge_index, num_nodes=x.size(0))

#         # Combine node and edge features
#         combined_features = torch.cat((x, aggregated_edge_attr), dim=1)
#         print("combined_features: ", combined_features.size())
#         # Project the combined features to the original node feature size
#         x = self.project(combined_features)

#         return x



# class GraphClassificationNetwork(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_classes, k=8):
#         super(GraphClassificationNetwork, self).__init__()
#         self.dfl1 = DynamicFilterLayer(in_channels, hidden_channels, k=k)
#         self.dfl2 = DynamicFilterLayer(hidden_channels, hidden_channels, k=k // 2)
#         self.dfbf = DynamicFilterBasedFeatures(hidden_channels, hidden_channels, k=k // 2)
#         # self.gc1 = GCNConv(hidden_channels * 2, hidden_channels)
#         self.gc1 = GCNConv(3, hidden_channels)
#         self.gc2 = GCNConv(hidden_channels, hidden_channels)
#         self.classifier = nn.Linear(hidden_channels, num_classes)

#     def forward(self, x, edge_index, batch):
#         x = self.dfl1(x, edge_index)
#         x = self.dfl2(x, edge_index)
#         x = self.dfbf(x, edge_index)

#         x = self.gc1(x, edge_index)
#         x = F.relu(x)
#         x = self.gc2(x, edge_index)
#         print(x.shape,batch.shape)
#         print("Shape of x:", x.shape)
#         print("Shape of batch tensor:", batch.shape)
#         print("Unique batches:", batch.unique().numel())
#         x = global_mean_pool(x, batch)  # Global mean pooling
#         return self.classifier(x)


# model = GraphClassificationNetwork(in_channels=3, hidden_channels=16, num_classes=10)


# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels, n_layers):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)  # for reproducibility

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(3, hidden_channels))  # First layer

#         for _ in range(n_layers - 2):
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))

#         self.convs.append(GCNConv(hidden_channels, 10))  # Last layer (10 classes in MNIST)

#     def forward(self, x, edge_index, batch):
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.5, training=self.training)

#         x = self.convs[-1](x, edge_index)
#         x = global_mean_pool(x, batch)  # Pool node features to get one per graph
#         return F.log_softmax(x, dim=1)



# model = GraphClassificationNetwork(in_channels=3, hidden_channels=16, num_classes=10)

# model = GCN(hidden_channels=16, n_layers=5)
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing, global_add_pool

class DynamicEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, classes, k, img_size):
        super(DynamicEdgeConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.classes = classes
        self.k = k
        self.img_size = img_size
        self.filters = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(classes)])
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.class_thresholds = np.linspace(0.5, 0.01, classes)  # Class-specific thresholds

    def forward(self, x, edge_index, pos, pixel_values):
        # Compute distance-based adjacency matrix
        A_dist = self.compute_adjacency_matrix(pos)

        # Compute pixel-based adjacency matrix
        A_pixel = self.compute_pixel_adjacency_matrix(pixel_values)

        filtered_x = [self.filters[c](x) for c in range(self.classes)]
        edge_features = []
        for c in range(self.classes):
            A_combined = self.combine_adjacency_matrices(A_dist, A_pixel, self.class_thresholds[c])
            edge_attr = self.compute_edge_features(filtered_x[c], edge_index, A_combined)
            edge_features.append(edge_attr)

        updated_x = torch.cat([x] + edge_features, dim=1)
        return updated_x

    def compute_adjacency_matrix(self, pos):
        coord = pos.cpu().numpy() / self.img_size
        dist = cdist(coord, coord)
        sigma = 0.05 * np.pi
        A = np.exp(-dist / sigma ** 2)
        return A

    def compute_pixel_adjacency_matrix(self, pixel_values):
        pixel_values_np = pixel_values.cpu().numpy() / 255.0
        dist_pixel = cdist(pixel_values_np, pixel_values_np)
        sigma = 0.05  # You might need to adjust this
        A_pixel = np.exp(-dist_pixel / sigma ** 2)
        return A_pixel

    def combine_adjacency_matrices(self, A_dist, A_pixel, threshold):
        # Convert A_dist and A_pixel to PyTorch tensors
        A_dist_tensor = torch.from_numpy(A_dist).float()
        A_pixel_tensor = torch.from_numpy(A_pixel).float()


        # min_dist = torch.min(A_dist_tensor)
        # max_dist = torch.max(A_dist_tensor)
        # mean_dist = torch.mean(A_dist_tensor)

        # # Calculate minimum, maximum, and mean values for A_pixel_tensor
        # min_pixel = torch.min(A_pixel_tensor)
        # max_pixel = torch.max(A_pixel_tensor)
        # mean_pixel = torch.mean(A_pixel_tensor)

        # # Print the results
        # print("Minimum value for A_dist_tensor:", min_dist.item())
        # print("Maximum value for A_dist_tensor:", max_dist.item())
        # print("Mean value for A_dist_tensor:", mean_dist.item())

        # print("Minimum value for A_pixel_tensor:", min_pixel.item())
        # print("Maximum value for A_pixel_tensor:", max_pixel.item())
        # print("Mean value for A_pixel_tensor:", mean_pixel.item())



        # Check if the model is on GPU and move adjacency matrices to the same device
        device = next(self.parameters()).device
        A_dist_tensor = A_dist_tensor.to(device)
        A_pixel_tensor = A_pixel_tensor.to(device)

        # Element-wise multiplication of the adjacency matrices
        A_combined = A_dist_tensor + A_pixel_tensor
        A_combined[A_combined < threshold] = 0

        return A_combined


    def compute_edge_features(self, x, edge_index, A_combined):
        row, col = edge_index
        edge_mask = A_combined[row, col] > 0
        filtered_edge_index = edge_index[:, edge_mask]
        edge_features = self.edge_mlp(torch.cat([x[filtered_edge_index[0]], x[filtered_edge_index[1]]], dim=1))
        aggregated_edge_features = global_add_pool(edge_features, filtered_edge_index[1], size=x.size(0))
        return aggregated_edge_features

class GraphClassificationModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_classes_dynamic, k, img_size):
        super(GraphClassificationModel, self).__init__()
        self.dynamic_edge_conv = DynamicEdgeConv(in_channels, hidden_channels, num_classes_dynamic, k, img_size)

        # Adjusted input channels based on the output of DynamicEdgeConv
        adjusted_in_channels = hidden_channels * (num_classes_dynamic + 1) + in_channels

        self.gcn1 = GCNConv(163, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, pos):
        # Use x as pixel values in DynamicEdgeConv
        x = self.dynamic_edge_conv(x, edge_index, pos, x)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out

num_nodes = 784 #testing for fixed graph
num_features = 3
num_classes = 10
num_classes_dynamic = 10
k = 8 
img_size = 32
# model = GraphClassificationModel(3, 16, num_classes, num_classes_dynamic, k=8)
model = GraphClassificationModel(3, 16, num_classes, num_classes_dynamic, k, img_size)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")
# Ensure CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    raise RuntimeError("CUDA is not available")


# device = torch.device('cpu')
# Move the model to GPU
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        batch = batch.to(device)  # Move batch to GPU
        optimizer.zero_grad()

        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.long)
        batch.y = batch.y.to(device)
        if batch.y.ndim > 1:
            batch.y = batch.y.squeeze(-1)

        # Include 'pos' in the model forward call
        out = model(batch.x, batch.edge_index, batch.batch, batch.pos)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.max(dim=1)[1]
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)

    return total_loss / len(train_loader), correct / total


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.long)
        batch.y = batch.y.to(device)
        if batch.y.ndim > 1:
            batch.y = batch.y.squeeze(-1)

        with torch.no_grad():
            # Include 'pos' in the model forward call
            out = model(batch.x, batch.edge_index, batch.batch, batch.pos)
            pred = out.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


for epoch in range(1, 21):  # e.g., 20 epochs
    loss, acc = train(model, train_loader, optimizer, criterion)
    print(loss,acc)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


