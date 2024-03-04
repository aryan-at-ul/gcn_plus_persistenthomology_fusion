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
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torchvision.transforms import ToTensor
import medmnist
from torch_geometric.utils import grid

if not os.path.exists('./data'):
    os.makedirs('./data')




import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torchvision.transforms import ToTensor
import medmnist
from torch_geometric.utils import grid
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torchvision.transforms import ToTensor
import medmnist
from torch_geometric.utils import grid
from focal_loss.focal_loss import FocalLoss

all_flags=['tissuemnist','pathmnist','chestmnist','dermamnist','octmnist','pnemoniamnist',
           'retinamnist','breastmnist','bloodmnist','tissuemnist','organamnist','organcmnist','organsmnist']

data_flag='breastmnist'
data_flag = 'OCTMNIST'
data_flag = data_flag.lower()
info = medmnist.INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])

class MedMNISTGraph(InMemoryDataset):
    def __init__(self, root, data_flag=data_flag, split='train', transform=None, download=True):
        self.info = medmnist.INFO[data_flag]
        self.n_channels = self.info['n_channels']
        self.DataClass = getattr(medmnist, self.info['python_class'])

        self.dataset = self.DataClass(root=root, split=split, transform=ToTensor(), download=download)

        super(MedMNISTGraph, self).__init__(root, transform, None, None)
        path = self.processed_paths[0] if split == 'train' else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ['train_graph.pt', 'test_graph.pt']

    def process(self):
        edge_index, pos = grid(28, 28)  # Assuming images are 28x28

        # Normalize pos by dividing by the image size
        normalized_pos = pos.float() / 28.0

        data_list = []
        for x, y in self.dataset:
            y_tensor = torch.tensor(y, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            # Reshape x and concatenate with normalized pos
            x = x.view(28*28, -1)  # Flatten and maintain channel information
            x_with_pos = torch.cat([x, normalized_pos], dim=1)  # Concatenate features with normalized coordinates

            data = Data(x=x_with_pos, edge_index=edge_index, pos=normalized_pos, y=y_tensor)
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
        self.gcn1 = GCNConv(323, 128)
        self.gcn2 = GCNConv(128, 128)
        self.classifier = nn.Linear(128, n_classes)
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
num_features = n_channels + 2
num_classes = 10
num_classes_dynamic = 10
model = GraphClassificationModel(num_features, 32, num_classes, num_classes_dynamic, k=8)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")
# Ensure CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise RuntimeError("CUDA is not available")


#LABEL PROPORTAION CALCULATION ONLY $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
import torchvision.transforms as transforms
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

is_multiclass= True if n_classes > 2 else False

DataClass = getattr(medmnist, info['python_class'])
print(n_classes,n_channels,data_flag,task,info)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset_imgs = DataClass(split='train', transform=data_transform, download=download)
test_dataset_imgs = DataClass(split='test', transform=data_transform, download=download)


def calculate_label_percentage(dataset, n_classes):
    label_counts = {i: 0 for i in range(n_classes)}  # Initialize counts for each label
    total_samples = len(dataset)
    return_percent = {}

    for x, y in dataset:
        label = y.item()  # Unpack the tuple and get the label
        label_counts[label] += 1

    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Label {label}: {percentage:.2f}%")
        return_percent[label] = percentage
    return return_percent.values()
        

print("Train Dataset Label Distribution:")
label_percentages = calculate_label_percentage(train_dataset_imgs, n_classes)
print("===================")
print("Test Dataset Label Distribution:")
test_label_percent = calculate_label_percentage(test_dataset_imgs, n_classes)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



# device = torch.device('cpu')
# Move the model to GPU
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001, amsgrad=True)

# this part is for imbalanced data @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# label_percentages = [34.35, 10.48, 7.95, 47.22]
n_classes = len(label_percentages)

# Convert percentages to proportions
label_proportions = [p / 100 for p in label_percentages]

# Calculate inverse and normalize
weights = [1.0 / p for p in label_proportions]
sum_weights = sum(weights)
normalized_weights = [n_classes * w / sum_weights for w in weights]

weights_tensor = torch.FloatTensor(normalized_weights)
print("Weights for Focal Loss:", weights_tensor)
weights_tensor = weights_tensor.to(device)

criterion = FocalLoss(gamma=0.5, weights=weights_tensor)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


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

        if is_multiclass:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)
        
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


for epoch in range(1, 51):  # e.g., 20 epochs
    loss, acc = train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


