
from shape_generator import draw_random_shape
from augmentations import random_augment,add_high_frequency_noise
from data_loaders import get_shape_loaders
from model import ResNet18, LinearProbe
import torch
import torch.nn as nn
import umap






def train_model(num_epochs, train_intermediate):
    train_loader, test_loader = get_shape_loaders(num_train=500, num_test=500, batch_size=64)
    shapes = ['circle', 'triangle', 'square', 'rectangle', 'ellipse']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=5)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bbox_criterion = nn.MSELoss()

    # Intermediate
    if train_intermediate:
        linear_probe = LinearProbe(input_dim=10, output_dim=1).to(device)
        inter_optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.001)
        inter_criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(num_epochs):
        for images, labels, bbox in train_loader:
            optimizer.zero_grad()
            images, labels, bbox = images.to(device), labels.to(device), bbox.to(device)
            
            # Forward pass
            intermediate, outputs, pred_bbox = model(images)
            
            loss = criterion(outputs, labels)
            bbox_loss = bbox_criterion(pred_bbox, bbox) * 100
            combined_loss = loss + bbox_loss
            combined_loss.backward()
            
            optimizer.step()
            
            # Intermediate predict
            if train_intermediate:
                inter_optimizer.zero_grad()
                intermediate = intermediate.detach()
                inter_outputs = linear_probe(intermediate)
                size = images.sum(dim=(1, 2, 3)) / 1000
                size = size.view(-1, 1)
                inter_loss = inter_criterion(inter_outputs, size)
                inter_loss.backward()
                inter_optimizer.step()
            
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Bbox Loss: {bbox_loss.item():.4f}')
        losses.append(loss.item())