from torch_geometric.data import InMemoryDataset, DataLoader
from torchvision.transforms import ToTensor
from torch_geometric.utils import grid
from medmnist.dataset import PathMNIST  # Adjust the import based on your setup
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torchvision.transforms import ToTensor
from torch_geometric.utils import grid
import medmnist  # Adjust this import if necessary
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.nn.conv import DynamicEdgeConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.nn.conv import DynamicEdgeConv


if not os.path.exists('./data'):
    os.makedirs('./data')


class MedMNISTGraph(InMemoryDataset):
    def __init__(self, root, data_flag='pathmnist', split='train', transform=None, download=True):
        self.info = medmnist.INFO[data_flag]
        self.DataClass = getattr(medmnist, self.info['python_class'])

        self.dataset = self.DataClass(root=root, split=split, transform=ToTensor(), download=download)

        super(MedMNISTGraph, self).__init__(root, transform)
        path = self.processed_paths[0] if split == 'train' else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ['train_graph.pt', 'test_graph.pt']

    def process(self):
        edge_index, pos = grid(28, 28)  # Assuming images are 28x28

        data_list = []
        for x, y in self.dataset:
            y_tensor = torch.tensor(y, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x=x.view(28*28, 3), edge_index=edge_index, pos=pos, y=y_tensor)  # Adjust for 3 channels
            data_list.append(data)

        if self.dataset.split == 'train':
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            torch.save(self.collate(data_list), self.processed_paths[1])

# Usage:
train_dataset = MedMNISTGraph('./data', split='train', download=True)
test_dataset = MedMNISTGraph('./data', split='test', download=True)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, MessagePassing

# class DynamicEdgeConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, classes, k):
#         super(DynamicEdgeConv, self).__init__(aggr='add')
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.classes = classes
#         self.k = k
#         self.filters = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(classes)])

#         self.edge_mlp = nn.Sequential(nn.Linear(2 * out_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))

#     def forward(self, x, edge_index):
#         filtered_x = [self.filters[c](x) for c in range(self.classes)]
#         edge_features = []
#         for c in range(self.classes):
#             edge_attr = self.compute_edge_features(filtered_x[c], edge_index)
#             edge_features.append(edge_attr)

#         updated_x = torch.cat([x] + edge_features, dim=1)
#         return updated_x

#     def compute_edge_features(self, x, edge_index):
#         row, col = edge_index
#         edge_features = self.edge_mlp(torch.cat([x[row], x[col]], dim=1))
#         aggregated_edge_features = global_add_pool(edge_features, col, size=x.size(0))
#         return aggregated_edge_features

# class GraphClassificationModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_classes, num_classes_dynamic, k):
#         super(GraphClassificationModel, self).__init__()
#         self.dynamic_edge_conv = DynamicEdgeConv(in_channels, hidden_channels, num_classes_dynamic, k)

#         adjusted_in_channels = hidden_channels * (num_classes_dynamic + 1)
#         # print("adjusted_in_channels: ", adjusted_in_channels)
#         self.gcn1 = GCNConv(323, hidden_channels)
#         self.gcn2 = GCNConv(hidden_channels, hidden_channels)
#         # self.gcn3 = GCNConv(hidden_channels, hidden_channels)
#         self.classifier = nn.Linear(hidden_channels, 10)#, bias=True)


#     def forward(self, x, edge_index, batch):
#         x = self.dynamic_edge_conv(x, edge_index)
#         # print("Shape of x after DynamicEdgeConv:", x.shape)
#         x = F.relu(self.gcn1(x, edge_index))
#         x = F.relu(self.gcn2(x, edge_index))
#         # x = F.relu(self.gcn3(x, edge_index))
#         x = global_add_pool(x, batch)
#         out = self.classifier(x)
#         # out = self.classifier2(x)
#         return out


# num_nodes = 784 #testing for fixed graph
# num_features = 3
# num_classes = 20
# num_classes_dynamic = 20
# model = GraphClassificationModel(3, 16, num_classes, num_classes_dynamic, k=8)
class DynamicEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, classes, k):
        super(DynamicEdgeConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.classes = classes
        self.k = k
        self.filters = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(classes)])
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        filtered_x = [self.filters[c](x) for c in range(self.classes)]
        edge_features = []
        for c in range(self.classes):
            edge_attr = self.compute_edge_features(filtered_x[c], edge_index)
            edge_features.append(edge_attr)

        updated_x = torch.cat([x] + edge_features, dim=1)
        return updated_x

    def compute_edge_features(self, x, edge_index):
        row, col = edge_index
        edge_features = self.edge_mlp(torch.cat([x[row], x[col]], dim=1))
        aggregated_edge_features = global_add_pool(edge_features, col, size=x.size(0))
        return aggregated_edge_features

class GraphClassificationModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_classes_dynamic, k):
        super(GraphClassificationModel, self).__init__()
        self.dynamic_edge_conv = DynamicEdgeConv(in_channels, hidden_channels, num_classes_dynamic, k)

        adjusted_in_channels = hidden_channels * (num_classes_dynamic + 1)
        # print("adjusted_in_channels: ", adjusted_in_channels)
        self.gcn1 = GCNConv(323, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        # self.gcn1 = GCNConv(hidden_channels * (num_classes_dynamic + 1), hidden_channels)
        # self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        # self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.dynamic_edge_conv(x, edge_index)
        # print("Shape of x after DynamicEdgeConv:", x.shape)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out


num_nodes = 784 #testing for fixed graph
num_features = 3
num_classes = 10
num_classes_dynamic = 10
model = GraphClassificationModel(3, 32, num_classes, num_classes_dynamic, k=8)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")
# Ensure CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
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
        # print(batch.x.shape, batch.edge_index.shape, batch.batch.shape)

        batch = batch.to(device)  # Move batch to GPU
        optimizer.zero_grad()

        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.long)
        batch.y = batch.y.to(device)
        if batch.y.ndim > 1:
            batch.y = batch.y.squeeze(-1)

        out = model(batch.x, batch.edge_index, batch.batch)
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
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


for epoch in range(1, 21):  # e.g., 20 epochs
    loss, acc = train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


