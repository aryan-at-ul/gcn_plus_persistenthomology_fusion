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
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import torch_geometric
import networkx as nx
from tqdm import tqdm 
all_flags=['tissuemnist','pathmnist','chestmnist','dermamnist','octmnist','pnemoniamnist',
           'retinamnist','breastmnist','bloodmnist','tissuemnist','organamnist','organcmnist','organsmnist']

data_flag='retinamnist'
# data_flag = 'OCTMNIST'
data_flag = data_flag.lower()
info = medmnist.INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])

if data_flag == 'retinamnist':
    train_transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.Lambda(lambda image: image.convert('RGB')),
    # torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5], std=[.5])
])
test_transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5], std=[.5])
])

# # load the data
# train_dataset = DataClass(split='train', transform=train_transform, download=download)
# test_dataset = DataClass(split='test', transform=test_transform, download=download)

pixel_threshold = 0.1

class MedMNISTGraph(InMemoryDataset):
    def __init__(self, root, data_flag=data_flag, split='train', transform=None, download=True):
        self.info = medmnist.INFO[data_flag]
        self.n_channels = self.info['n_channels']
        self.split = split
        self.DataClass = getattr(medmnist, self.info['python_class'])

        # self.dataset = self.DataClass(root=root, split=split, transform=ToTensor(), download=download)
        if split == 'train':
            self.dataset = self.DataClass(root=root, split=split, transform=train_transform, download=download)
        else:
            self.dataset = self.DataClass(root=root, split=split, transform=test_transform, download=download)

        super(MedMNISTGraph, self).__init__(root, transform, None, None)
        path = self.processed_paths[0] if split == 'train' else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        

    @property
    def processed_file_names(self):
        return ['train_graph.pt', 'test_graph.pt']

    def process(self):

        
        processed_file_path = self.processed_paths[0] if self.split == 'train' else self.processed_paths[1]

        # Check if the processed file already exists
        if os.path.isfile(processed_file_path):
            print("Processed file already exists. Loading data.")
            self.data, self.slices = torch.load(processed_file_path)
            return 

        edge_index, pos = grid(28, 28)  # Assuming images are 28x28
        pixel_threshold = 0.1
        n_hop = 2
        # Normalize pos by dividing by the image size
        normalized_pos = pos.float() / 28.0

        data_list = []
        for x, y in tqdm(self.dataset, desc="Processing Graphs"):
            removed_edges = set()
            y_tensor = torch.tensor(y, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = x / x.max()
            x = x.view(28*28, -1)  # Flatten and maintain channel information
            num_nodes = x.shape[0]
            x_with_pos = torch.cat([x, normalized_pos], dim=1)  # Concatenate features with normalized coordinates
            y_tensor = torch.tensor(y, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, pos=normalized_pos, y=y_tensor)
            graph = torch_geometric.utils.to_networkx(data, to_undirected=True)


            for node in graph.nodes():
                path_lengths = nx.single_source_shortest_path_length(graph, node)
                neighbors = [neighbor for neighbor, length in path_lengths.items() if length <= n_hop]
                node_pixel_value = data.x[node].mean()
                neighbor_pixel_values = data.x[neighbors].mean(dim=0)

                for neighbor in neighbors:
                    if (node, neighbor) not in removed_edges and graph.has_edge(node, neighbor):
                        neighbor_pixel_value = data.x[neighbor].mean()
                        pixel_difference = abs(node_pixel_value - neighbor_pixel_value)

                        if pixel_difference > pixel_threshold:
                            graph.remove_edge(node, neighbor)
                            removed_edges.add((node, neighbor))
                            removed_edges.add((neighbor, node))
                        elif pixel_difference <= pixel_threshold:
                            graph.nodes[neighbor]['x'] = neighbor_pixel_values.numpy()

            updated_x = torch.tensor([graph.nodes[node]['x'] for node in graph.nodes()], dtype=torch.float)
            new_data = torch_geometric.utils.from_networkx(graph)
            new_data.x = torch.cat([updated_x, normalized_pos], dim=1)
            new_data.y = torch.tensor(y, dtype=torch.long)
            new_data.pos = normalized_pos

            data_list.append(new_data)
            updated_edge_index = torch_geometric.utils.from_networkx(graph)
            data.edge_index = updated_edge_index

        if self.dataset.split == 'train':
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            torch.save(self.collate(data_list), self.processed_paths[1])

# Usage:

download = False
DataClass = getattr(medmnist, info['python_class'])
train_dataset_imgs = DataClass(split='train', transform=train_transform, download=download)
test_dataset_imgs = DataClass(split='test', transform=test_transform, download=download)



train_dataset = MedMNISTGraph('./data', split='train', download=True)
test_dataset = MedMNISTGraph('./data', split='test', download=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_cluster import knn_graph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import torch_geometric.utils
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, coalesce
from torch_cluster import knn_graph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def add_n_hop_edges(edge_index, num_nodes, n):
    # Convert to dense adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

    # Add edges for n-hop neighbors
    adj_n = adj.clone()
    for _ in range(n - 1):
        adj_n = torch.matmul(adj_n, adj)

    # Convert back to edge index format, ensuring no self-loops or duplicates
    adj_n = torch.where(adj_n > 0, 1, 0)
    adj_n.fill_diagonal_(0)
    new_edge_index, _ = dense_to_sparse(adj_n)
    
    return new_edge_index

# def merge_nodes(x, edge_index, mask):
#     # Step 1: Identify nodes to merge
#     nodes_to_merge = mask.nonzero(as_tuple=True)[0]
#     print("Nodes to merge:", nodes_to_merge, nodes_to_merge.shape)  # Debugging print

#     # Step 2: Create new nodes
#     if nodes_to_merge.numel() > 0:
#         merged_node_features = x[nodes_to_merge].mean(dim=0, keepdim=True)
#         new_x = torch.cat([x, merged_node_features], dim=0)
#         new_node_index = new_x.size(0) - 1  # Index of the new merged node
#     else:
#         new_x = x.clone()
#         new_node_index = None

#     print("New node index:", new_node_index)  # Debugging print

#     # Step 3: Update edge index
#     new_edge_index_list = []
#     for i in range(edge_index.size(1)):
#         src, dest = edge_index[:, i].tolist()
#         if new_node_index is not None:
#             src = new_node_index if src in nodes_to_merge else src
#             dest = new_node_index if dest in nodes_to_merge else dest
#         if src != dest and src < new_x.size(0) and dest < new_x.size(0):  # Avoid self-loops and ensure indices are within bounds
#             new_edge_index_list.append([src, dest])

#     if not new_edge_index_list:
#         print("Warning: No edges in new_edge_index after merging.")
#         return new_x, torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

#     new_edge_index = torch.tensor(new_edge_index_list, dtype=torch.long, device=edge_index.device).t().contiguous()

#     print("Sample of new_edge_index:", new_edge_index[:, :10])  # Debugging print

#     # Step 4: Remove duplicates and self-loops
#     new_edge_index, _ = torch_geometric.utils.coalesce(new_edge_index, None, num_nodes=new_x.size(0))

#     print("Sample of new_edge_index after coalesce:", new_edge_index[:, :10])  # Debugging print

#     return new_x, new_edge_index
# def merge_nodes(x, edge_index, merge_candidates):
#     # Create a mapping for merged nodes
#     merge_mapping = {}
#     new_x_list = []
#     for node, candidates in merge_candidates.items():
#         if node not in merge_mapping:
#             # Compute the mean features for the group
#             group_nodes = torch.tensor(list(candidates), device=x.device)
#             mean_features = x[group_nodes].mean(dim=0)
#             new_index = len(new_x_list)
#             new_x_list.append(mean_features)

#             # Update merge mapping
#             for n in candidates:
#                 merge_mapping[n] = new_index

#     # Create new feature matrix
#     new_x = torch.stack(new_x_list)

#     # Update edge index
#     new_edge_index_list = []
#     for i in range(edge_index.size(1)):
#         src, dest = edge_index[:, i].tolist()
#         new_src = merge_mapping.get(src, src)
#         new_dest = merge_mapping.get(dest, dest)
#         if new_src != new_dest:
#             new_edge_index_list.append([new_src, new_dest])

#     if len(new_edge_index_list) > 0:
#         new_edge_index = torch.tensor(new_edge_index_list, dtype=torch.long, device=edge_index.device).t().contiguous()
#     else:
#         new_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)


#     edge_set = {tuple(item) for item in new_edge_index.t().tolist()}

#     # Remove self-loops
#     edge_set = {edge for edge in edge_set if edge[0] != edge[1]}

#     # Convert back to tensor
#     if edge_set:
#         new_edge_index = torch.tensor(list(edge_set), dtype=torch.long, device=edge_index.device).t().contiguous()
#     else:
#         new_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

#     max_edge_index = new_edge_index.max().item()
#     assert max_edge_index < new_x.size(0), f"New edge index out of bounds: {max_edge_index} >= {new_x.size(0)}"

#     # new_edge_index = torch.tensor(new_edge_index_list, dtype=torch.long, device=edge_index.device).t().contiguous()
#     print("new edge before coalesce:", new_edge_index.shape) 
#     # new_edge_index, _ = torch_geometric.utils.coalesce(new_edge_index, None, num_nodes=new_x.size(0))
#     print("new edge ater coalesce:", new_edge_index.shape)  # Debugging print
#     return new_x, new_edge_index
# def merge_nodes(x, edge_index, merge_candidates):
#     # Step 1: Create a copy of x to modify
#     new_x = x.clone()
    
#     # Step 2: Replace features of nodes in each merge group with their mean
#     for _, group in merge_candidates.items():
#         # Compute mean feature of the group
#         mean_feature = x[list(group)].mean(dim=0)
#         for node in group:
#             new_x[node] = mean_feature

#     # Step 3: Update the edge_index to reflect merged nodes
#     new_edge_index_list = []
#     for src, dest in edge_index.t().tolist():
#         # Replace src and dest with their corresponding merged nodes
#         src = next(iter(merge_candidates.get(src, {src})))
#         dest = next(iter(merge_candidates.get(dest, {dest})))
#         if src != dest:
#             new_edge_index_list.append([src, dest])

#     new_edge_index = torch.tensor(new_edge_index_list, dtype=torch.long, device=edge_index.device).t().contiguous()

#     # Debugging
#     print("Updated edge_index:", new_edge_index)
#     print("Max index in updated edge_index:", new_edge_index.max().item())

#     # Check if new_edge_index is within bounds
#     assert new_edge_index.max().item() < new_x.size(0), "Edge index out of bounds"

#     return new_x, new_edge_index
def merge_nodes(x, edge_index, merge_candidates):
    # Create new features list and node mapping
    new_features = []
    node_mapping = {}

    new_index = 0
    for node in range(x.size(0)):
        if node in merge_candidates:
            # Process merge candidates
            if node not in node_mapping:
                group = merge_candidates[node]
                group_features = x[list(group)].mean(dim=0)
                new_features.append(group_features)

                for member in group:
                    node_mapping[member] = new_index

                new_index += 1
        else:
            # Keep original features for non-merged nodes
            new_features.append(x[node])
            node_mapping[node] = new_index
            new_index += 1

    # Create new feature matrix
    new_x = torch.stack(new_features)

    # Update edge index
    new_edge_index = []
    for src, dest in edge_index.t().tolist():
        new_src = node_mapping.get(src, -1)  # Debugging
        new_dest = node_mapping.get(dest, -1)  # Debugging
        if new_src == -1 or new_dest == -1:
            print(f"Error: Missing node in mapping. Source: {src}, Dest: {dest}")  # Debugging
            continue
        new_edge_index.append([new_src, new_dest])

    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long, device=edge_index.device).t().contiguous()

    return new_x, new_edge_index





def relabel_targets(y):
    unique_labels = torch.unique(y)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    new_y = torch.tensor([label_mapping[label.item()] for label in y], dtype=y.dtype, device=y.device)
    return new_y

def custom_pooling(batch_data, n_hop=2, threshold=0.3):
    x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
    y = batch_data.y if hasattr(batch_data, 'y') else None
    if y is not None:
        y = relabel_targets(y)

    # Add n-hop edges
    num_nodes = x.size(0)
    extended_edge_index = add_n_hop_edges(edge_index, num_nodes, n_hop)

    # Identify merge candidates
    merge_candidates = {}
    for i in range(extended_edge_index.size(1)):
        src, dest = extended_edge_index[:, i].tolist()
        if torch.norm(x[src] - x[dest]) < threshold:
            if src not in merge_candidates:
                merge_candidates[src] = set()
            merge_candidates[src].add(dest)

    print("Number of merge candidates:", len(merge_candidates))  # Diagnostic print

    if not merge_candidates:
        print("No node merging performed.")
        return batch_data

    # Merge nodes and update the graph structure
    new_x, new_edge_index = merge_nodes(x, edge_index, merge_candidates)

    print("New x size after merging:", new_x.size(0))  # Diagnostic print

    # Update batch tensor
    if new_x.size(0) != x.size(0):
        new_batch_size = new_x.size(0) - x.size(0)
        new_batch = torch.cat([batch, batch[-1].repeat(new_batch_size)])
    else:
        new_batch = batch

    return Data(x=new_x, edge_index=new_edge_index, batch=new_batch, y=y)


# def custom_pooling(batch_data, n_hop=2, threshold=0.3):
#     x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
#     y = batch_data.y if hasattr(batch_data, 'y') else None
#     # y = relabel_targets(batch_data.y)
#     # Only relabel targets if y is not None
#     if y is not None:
#         y = relabel_targets(y)
#     # Add n-hop edges
#     num_nodes = x.size(0)
#     extended_edge_index = add_n_hop_edges(edge_index, num_nodes, n_hop)

#     # Identify merge candidates
#     merge_candidates = {}
#     for i in range(extended_edge_index.size(1)):
#         src, dest = extended_edge_index[:, i].tolist()
#         if torch.norm(x[src] - x[dest]) < threshold:
#             if src not in merge_candidates:
#                 merge_candidates[src] = set()
#             merge_candidates[src].add(dest)

#     if not merge_candidates:
#         print("No node merging performed.")
#         return batch_data

#     # Merge nodes and update the graph structure
#     new_x, new_edge_index = merge_nodes(x, edge_index, merge_candidates)

#     # Update batch tensor
#     if new_x.size(0) != x.size(0):
#         new_batch_size = new_x.size(0) - x.size(0)
#         new_batch = torch.cat([batch, batch[-1].repeat(new_batch_size)])
#     else:
#         new_batch = batch

#     return Data(x=new_x, edge_index=new_edge_index, batch=new_batch, y=y)


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        assert edge_index.size(0) == 2, "Edge index must have a shape of [2, num_edges]"
        print("Max edge index:", edge_index.max().item(), "Number of nodes:", x.size(0))
        assert edge_index.max().item() < x.size(0), "Edge index out of bounds"

        print("Input edge_index shape:", edge_index.shape)
        print("Input x shape:", x.shape)
        # First GCN layer
        print("Before GCN1 - edge_index shape:", edge_index.shape)  # Diagnostic print
        x = F.relu(self.conv1(x, edge_index))
        print("Post-GCN1 x shape:", x.shape)  # Diagnostic print
        x = F.dropout(x, training=self.training)

        # First custom pooling
        pooled_data = custom_pooling(Batch(x=x, edge_index=edge_index, batch=batch))
        assert pooled_data.edge_index.max().item() < pooled_data.x.size(0), "Edge index out of bounds after pooling"

        x, edge_index, batch = pooled_data.x, pooled_data.edge_index, pooled_data.batch
        print("After pooling - edge_index shape:", edge_index.shape,x.shape)  # Diagnostic print
        # First Linear layer
        x = F.relu(self.lin1(x))

        # Second GCN layer
        print("Before GCN2 - edge_index shape:", edge_index.shape)  # Diagnostic print

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)

        # Second custom pooling
        # pooled_data = custom_pooling(Batch(x=x, edge_index=edge_index, batch=batch))
        # x, edge_index, batch = pooled_data.x, pooled_data.edge_index, pooled_data.batch

        # Second Linear layer
        x = F.relu(self.lin2(x))

        # Global mean pooling
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # Classifier
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

# Custom pooling function and other necessary functions go here (as defined earlier)

# Example usage
# model = GCNModel(in_channels=3, hidden_channels=64, out_channels=128, num_classes=10)
# data = Data(x=torch.randn(16, 3), edge_index=torch.randint(0, 16, (2, 30)))

# # Forward pass through the model
# out = model(data)



num_nodes = 784 #testing for fixed graph
num_features = n_channels + 2
num_classes = 10
num_classes_dynamic = 10
model = GCNModel(in_channels=5, hidden_channels=64, out_channels=128, num_classes=n_classes)#GCNModel(3, 32, num_classes, num_classes_dynamic, k=8)

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

class GCN_x(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN_x, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128,128)
        self.conv4 = GCNConv(128, 64)
        self.lin1 = Linear(64, 32)
        #self.lin2 = Linear(128,64)
        self.lin = Linear(32, 9)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin(x)
        
        return x

# model = GCN_x(hidden_channels=16)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
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

criterion = FocalLoss(gamma=0, weights=weights_tensor)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# criterion = torch.nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        print(batch.x.shape, batch.edge_index.shape, batch.batch.shape)
        print("Max node index in edge_index:", batch.edge_index.max().item())
        print("Node feature matrix size:", batch.x.size(0))
        batch = batch.to(device)  # Move batch to GPU
        optimizer.zero_grad()

        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.long)
        batch.y = batch.y.to(device)
        if batch.y.ndim > 1:
            batch.y = batch.y.squeeze(-1)

        out = model(batch)

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
            out = model(batch)
            pred = out.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


for epoch in range(1, 51):  # e.g., 20 epochs
    loss, acc = train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


