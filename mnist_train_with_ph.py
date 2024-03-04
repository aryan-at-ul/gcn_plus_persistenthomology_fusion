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
from tqdm import tqdm
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, DynamicEdgeConv
import numpy as np
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch
from torch_geometric.data import InMemoryDataset, Data
from skimage.segmentation import slic
from skimage.future import graph
import numpy as np
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
from torch_geometric.nn import GCNConv, EdgePooling
# from giotto_tda import CubicalPersistence, BettiCurve
from sklearn.datasets import fetch_openml
from gtda.plotting import plot_heatmap
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgePooling
from torch_geometric.data import Batch
import matplotlib.pyplot as plt




if not os.path.exists('./data'):
    os.makedirs('./data')



all_flags=['tissuemnist','pathmnist','chestmnist','dermamnist','octmnist','pnemoniamnist',
           'retinamnist','breastmnist','bloodmnist','tissuemnist','organamnist','organcmnist','organsmnist']

data_flag='breastmnist'
# data_flag = 'OCTMNIST'
data_flag = data_flag.lower()
info = medmnist.INFO[data_flag]
n_channels = info['n_channels']
n_classes = 10#len(info['label'])

if data_flag == 'breastmnist':
    train_transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.Lambda(lambda image: image.convert('RGB')),
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])



import torch
# from torch_geometric.datasets import InMemoryDataset
from torch_geometric.data import Data
from torchvision.datasets import CIFAR10
from torchvision import transforms
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# from skimage import graph
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from skimage.segmentation import slic
from skimage.future import graph
from tqdm import tqdm
import numpy as np

class MNISTGraph(InMemoryDataset):
    def __init__(self, root, is_train=True, transform=None, pre_transform=None, download=True):
        self.is_train = is_train
        self.dataset = MNIST(root=root, train=is_train, transform=transforms.ToTensor(), download=download)
        super(MNISTGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0] if is_train else self.processed_paths[1])

    @property
    def processed_file_names(self):
        return ['mnist_train_graph.pt', 'mnist_test_graph.pt']

    def process(self):
        data_list = []
        for x, y in tqdm(self.dataset, desc="Processing Data"):
            image = x.numpy().squeeze()
            segments = slic(image, n_segments=300, compactness=10, sigma=1)

            # Calculate centroids and average intensities for superpixels
            centroids = self.calculate_superpixel_centroids(segments)
            features = self.calculate_superpixel_features(image, segments)

            # Extract edge_index from the RAG
            rag = graph.rag_mean_color(image, segments, mode='similarity')
            edge_index = self.extract_edge_index_from_rag(rag)

            # Create Data object
            data = Data(x=torch.tensor(features, dtype=torch.float),
                        pos=torch.tensor(centroids, dtype=torch.float),
                        edge_index=edge_index,
                        y=torch.tensor(y, dtype=torch.long),
                        image=torch.tensor(image, dtype=torch.float).unsqueeze(0))
            data_list.append(data)

        # Save processed data
        torch.save(self.collate(data_list), self.processed_paths[0] if self.is_train else self.processed_paths[1])

    def calculate_superpixel_centroids(self, segments):
        centroids = []
        for segVal in np.unique(segments):
            mask = segments == segVal
            centroid = np.mean(np.column_stack(np.where(mask)), axis=0)
            centroids.append(centroid)
        return centroids

    def calculate_superpixel_features(self, image, segments):
        features = []
        for segVal in np.unique(segments):
            mask = segments == segVal
            average_intensity = np.mean(image[mask])
            features.append([average_intensity])
        return features

    def extract_edge_index_from_rag(self, rag):
        edge_index = []
        for edge in rag.edges():
            src, dest = edge
            edge_index.append([src, dest])

        return torch.tensor(edge_index, dtype=torch.long).t()



def custom_collate(data_list):
    data_list = [data for data in data_list if data is not None]  # Filter out None entries
    x_list, edge_index_list, y_list, pos_list, image_list = zip(*data_list)

    # Stack the tensors along the batch dimension
    x = torch.stack(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    y = torch.stack(y_list, dim=0)
    pos = torch.stack(pos_list, dim=0)

    # Stack the image tensors along the batch and channel dimensions
    image = torch.stack([img.unsqueeze(0) for img in image_list], dim=0)
    image = image.view(-1, *image.size()[2:])  # Flatten the batch and channel dimensions

    # Create a DataBatch object
    batch = torch.arange(len(data_list), dtype=torch.long)
    ptr = torch.zeros(len(data_list) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(torch.tensor([x.size(0) for x in x_list]), dim=0)

    data_batch = Batch(x=x, edge_index=edge_index, y=y, pos=pos, image=image, batch=batch, ptr=ptr)

    return data_batch


train_dataset = MNISTGraph('./data', is_train=True, download=True)
test_dataset = MNISTGraph('./data', is_train=False, download=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
print(train_dataset[0])





class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)  # for reproducibility
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, n_classes)  

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pool node features to get one per graph
        return F.log_softmax(x, dim=1)

model = GCN(hidden_channels=16)


class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        return x

class PersistentHomologyLayer(nn.Module):
    def __init__(self):
        super(PersistentHomologyLayer, self).__init__()
        self.filtration = CubicalPersistence()
        self.vectorizer = BettiCurve()

    def forward(self, batch_images):
        diagram_vectors = []
        # print("in persistent")
        for img in batch_images:
            # print("in persistent homo class image shape:",img.shape)
            # img_2d = img.view(1, -1).cpu().numpy()
            img_2d = img.view(img.size(0), -1).cpu().numpy()
            # print("in persistent homo class image 2d shape:",img.shape)
            diagram = self.filtration.fit_transform(img_2d)
            diagram_vector = self.vectorizer.fit_transform(diagram)
            # print("in persistent homo class diagram vector shape:",diagram_vector.shape)
            diagram_vectors.append(torch.tensor(diagram_vector, dtype=torch.float))
        # print("size of persitent diagrams this should be 16/batch  size:",len(diagram_vectors))
        # print("out persistent")
        return torch.stack(diagram_vectors)




class AttentionLayer(nn.Module):
    def __init__(self, node_feature_size, diagram_feature_size):
        super(AttentionLayer, self).__init__()
        self.node_feature_transform = nn.Linear(node_feature_size, diagram_feature_size)
        self.attention = nn.Linear(diagram_feature_size, 1)

    def forward(self, node_features, diagram_vector, batch):
        # Transform node features to have the same size as diagram features
        transformed_node_features = self.node_feature_transform(node_features)

        # Compute attention scores
        attention_scores = self.attention(transformed_node_features + diagram_vector[batch])
        attention_scores = F.softmax(attention_scores, dim=0)

        # Apply attention
        attended_features = attention_scores * transformed_node_features

        return attended_features + node_features  # Combine attended features with original node features




class FeatureUpdateLayer(nn.Module):
    def __init__(self, in_channels, out_channels, diagram_vector_size):
        super(FeatureUpdateLayer, self).__init__()
        self.lin = nn.Linear(in_channels + diagram_vector_size, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, diagram_vector, batch):
        node_counts = torch.bincount(batch).to(x.device)
        diagram_vector = diagram_vector.to(x.device)
        # print("x shape and diagram shape:",x.shape, diagram_vector.shape)
        node_counts = torch.bincount(batch)
        repeated_diagram_vector = torch.repeat_interleave(diagram_vector, node_counts, dim=0)
        # print("x shape and diagram shape next step:",x.shape, repeated_diagram_vector.shape)
        # repeated_diagram_vector = diagram_vector.repeat_interleave(torch.bincount(batch))
        # print("x shape and diagram shape next step:",x.shape, repeated_diagram_vector.shape)
        x = torch.cat([x, repeated_diagram_vector], dim=-1)
        x = self.lin(x)
        x = self.relu(x)
        return x


# class FeatureUpdateLayer(nn.Module):
#     def __init__(self, in_channels, out_channels = 64, diagram_vector_size = 600):
#         super(FeatureUpdateLayer, self).__init__()
#         self.query = nn.Linear(in_channels, out_channels)
#         self.key = nn.Linear(diagram_vector_size, out_channels)
#         self.value = nn.Linear(diagram_vector_size, out_channels)
#         self.out = nn.Linear(out_channels, out_channels)
#         self.layer_norm = nn.LayerNorm(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x, diagram_vector, batch):
#         node_counts = torch.bincount(batch).to(x.device)
#         # node_counts = torch.bincount(batch)
#         diagram_vector = diagram_vector.to(x.device)
#         repeated_diagram_vector = torch.repeat_interleave(diagram_vector, node_counts, dim=0)

#         # Transform to query, key, value
#         q = self.query(x)  # Query from node features
#         k = self.key(repeated_diagram_vector)  # Key from diagram vector
#         v = self.value(repeated_diagram_vector)  # Value from diagram vector

#         # Scaled Dot-Product Attention
#         attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (64 ** 0.5), dim=-1)
#         attn_output = torch.matmul(attn_weights, v)

#         # Combine attended output with original x
#         combined_output = self.out(attn_output) + x
#         x = self.layer_norm(combined_output)
#         x = self.relu(combined_output)
#         print("x shape:",x.shape)
#         return x



class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.lin = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.lin(x)
        return x


class MultiScaleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_size):
        super(MultiScaleGNN, self).__init__()

        # Define each layer separately
        self.gnn_layer = GNNLayer(in_channels, hidden_channels)
        self.gnn_layer_2 = GNNLayer(hidden_channels, hidden_channels)
        self.ph_layer = PersistentHomologyLayer()
        self.update_layer = FeatureUpdateLayer(hidden_channels, hidden_channels, diagram_vector_size=600)
        self.edge_pool = EdgePooling(hidden_channels)
        self.classifier = Classifier(hidden_channels, num_classes)
        self.pool_size = pool_size


    def forward(self, data):
        # print("data", data)

        def visualize_batch_image(batch_img, title):
            img = batch_img[0].cpu().numpy()  # Convert the first image of the batch to NumPy array
            img = np.transpose(img, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to [0, 1]
            img_min = img.min()
            img_max = img.max()
            img = (img - img_min) / (img_max - img_min)

            plt.imshow(img)
            plt.title(title)
            plt.show()


        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = data.image.shape[0] // 3  # Assuming the number of channels is 3
        batch_images = data.image.view(batch_size, 1, 28, 28)
        # visualize_batch_image(batch_images, "Original batch images")
        # First cycle of operations
        x = self.gnn_layer(x, edge_index)
        diagram_vector = self.ph_layer(batch_images)
        diagram_vector = diagram_vector.view(diagram_vector.size(0), -1)
        batch_images = F.avg_pool2d(batch_images, kernel_size=2)
        x = self.update_layer(x, diagram_vector, batch)
        x, edge_index, batch, _ = self.edge_pool(x, edge_index, batch)
        # visualize_batch_image(batch_images, "after first cycle")
        # Second cycle of operations
        x = self.gnn_layer_2(x, edge_index)
        diagram_vector = self.ph_layer(batch_images)  # Use updated batch_images
        diagram_vector = diagram_vector.view(diagram_vector.size(0), -1)
        batch_images = F.avg_pool2d(batch_images, kernel_size=2)
        x = self.update_layer(x, diagram_vector, batch)
        x, edge_index, batch, _ = self.edge_pool(x, edge_index, batch)

        # visualize_batch_image(batch_images, "after second cycle")
        # Third cycle of operations
        x = self.gnn_layer_2(x, edge_index)
        diagram_vector = self.ph_layer(batch_images)
        diagram_vector = diagram_vector.view(diagram_vector.size(0), -1)
        batch_images = F.avg_pool2d(batch_images, kernel_size=2)
        x = self.update_layer(x, diagram_vector, batch)
        x, edge_index, batch, _ = self.edge_pool(x, edge_index, batch)

        # visualize_batch_image(batch_images, "after last cycle")
        batch_images = batch_images.view(-1, *batch_images.shape[2:])
        data.image = batch_images

        # Global pooling and classification
        graph_x = global_mean_pool(x, batch)
        out = self.classifier(graph_x)

        return out



model = MultiScaleGNN(in_channels=3, hidden_channels=64, num_classes=10, pool_size=2)


print(model)

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
BATCH_SIZE = 64
lr = 0.001
# task = info['task']
# n_channels = info['n_channels']
# n_classes = len(info['label'])

is_multiclass= True if n_classes > 2 else False

# DataClass = getattr(medmnist, info['python_class'])
# print(n_classes,n_channels,data_flag,task,info)

# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

# # load the data
# train_dataset_imgs = DataClass(split='train', transform=data_transform, download=download)
# test_dataset_imgs = DataClass(split='test', transform=data_transform, download=download)


# def calculate_label_percentage(dataset, n_classes):
#     label_counts = {i: 0 for i in range(n_classes)}  # Initialize counts for each label
#     total_samples = len(dataset)
#     return_percent = {}

#     for x, y in dataset:
#         label = y.item()  # Unpack the tuple and get the label
#         label_counts[label] += 1

#     for label, count in label_counts.items():
#         percentage = (count / total_samples) * 100
#         print(f"Label {label}: {percentage:.2f}%")
#         return_percent[label] = percentage
#     return return_percent.values()
        

# print("Train Dataset Label Distribution:")
# label_percentages = calculate_label_percentage(train_dataset_imgs, n_classes)
# print("===================")
# print("Test Dataset Label Distribution:")
# test_label_percent = calculate_label_percentage(test_dataset_imgs, n_classes)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# model = GCN(hidden_channels=16)
# model = GCN_x(hidden_channels=16)
# model = GCN()
# device = torch.device('cpu')
device = torch.device('cuda')
# Move the model to GPU
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay = 0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001, amsgrad=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)


# this part is for imbalanced data @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# label_percentages = [34.35, 10.48, 7.95, 47.22]
# n_classes = len(label_percentages)

# # Convert percentages to proportions
# label_proportions = [p / 100 for p in label_percentages]

# # Calculate inverse and normalize
# weights = [1.0 / p for p in label_proportions]
# sum_weights = sum(weights)
# normalized_weights = [n_classes * w / sum_weights for w in weights]
# weights_tensor = torch.FloatTensor(normalized_weights)
# print("Weights for Focal Loss:", weights_tensor)
# weights_tensor = weights_tensor.to(device)

# criterion = FocalLoss(gamma=1, weights=weights_tensor)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
criterion = torch.nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        # print(batch)
        # batch.edge_index = batch.edge_index - 1 
        # print("data attributes x shape, edge shape, batch shpae, batch y shape:",batch.x.shape, batch.edge_index.shape, batch.batch.shape, batch.y.shape)
        # print(batch.edge_index.max(), batch.x.size(0))

        batch = batch.to(device)  # Move batch to GPU
        optimizer.zero_grad()

        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.long)
        batch.y = batch.y.to(device)
        if batch.y.ndim > 1:
            batch.y = batch.y.squeeze(-1)

        # out = model(batch.x, batch.edge_index, batch.batch)
        out = model(batch)

        if is_multiclass:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)
        
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f" Loss (should be decreasing): {loss.item()}")
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
            # out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


for epoch in range(1, 101):  
    loss, acc = train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    scheduler.step(loss)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
