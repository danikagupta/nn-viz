import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchviz import make_dot
import streamlit as st
from io import BytesIO
import tempfile
import os
import graphviz

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        
        # Initialize weights and biases to zero
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc1.bias, a=-0.1, b=0.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to plot and return a matplotlib figure
def plot_loss(losses):
    fig, ax = plt.subplots()
    ax.plot(range(len(losses)), losses, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.legend()
    return fig

# Function to visualize the neural network architecture with weights and biases on edges
def visualize_architecture_with_weights(model):
    dot = graphviz.Digraph()
    
    # Input layer nodes
    dot.node('X1', 'Input 1')
    dot.node('X2', 'Input 2')

    # Hidden layer nodes
    for i in range(4):
        bias = model.fc1.bias[i].detach().item()
        dot.node(f'H{i+1}', f'Hidden {i+1} Bias: {bias:.2f}')
    
    # Output layer node
    output_bias = model.fc2.bias[0].detach().item()
    dot.node('Y', f'Output Bias: {output_bias:.2f}')

    # Connect input to hidden layer with weights on edges
    for i in range(2):
        for j in range(4):
            weight = model.fc1.weight[j, i].detach().item()
            dot.edge(f'X{i+1}', f'H{j+1}', label=f'w={weight:.2f}')

    # Connect hidden to output layer with weights on edges
    for i in range(4):
        weight = model.fc2.weight[0, i].detach().item()
        dot.edge(f'H{i+1}', 'Y', label=f'w={weight:.2f}')

    return dot

# Streamlit app
def main():
    st.title("Visualizing Neural Network Training with PyTorch and Streamlit")

    # Reinitialize weights button
    if 'model' not in st.session_state:
        st.session_state.model = None

    # User input for training data using Streamlit data editor
    st.sidebar.subheader("Input Training Data")
    data = st.sidebar.data_editor(
        pd.DataFrame(
            {'Feature 1': [1.0, 2.0, 3.0], 'Feature 2': [4.0, 5.0, 6.0], 'Label': [7.0, 8.0, 9.0]}
        ),
        num_rows="dynamic",
        key="training_data_editor"
    )

    X_data = data[['Feature 1', 'Feature 2']].values.tolist()
    y_data = data[['Label']].values.tolist()

    X = torch.tensor(X_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)

    # Create model, loss, and optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()
    learning_rate = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.1, step=0.01)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10
    losses = []

    # Training the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss for visualization
        losses.append(loss.item())

        # Update weights and visualize after training step
        st.subheader(f"Model Architecture at Epoch {epoch+1}")
        if os.system("which dot") == 0:  # Check if Graphviz is installed
            dot = visualize_architecture_with_weights(model)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                dot.render(tmpfile.name, format='png')
                st.image(tmpfile.name + ".png", caption=f'Model Architecture with Weights and Biases at Epoch {epoch+1}', use_container_width=True)
        else:
            st.warning("Graphviz is not installed or not found in PATH. Please install Graphviz to visualize the model architecture.")

        # Display loss value at each epoch
        st.write(f"### Epoch {epoch+1}")
        st.write(f"**Loss:** {loss.item():.4f}")

    # Plot the loss
    st.subheader("Training Loss")
    loss_fig = plot_loss(losses)
    st.pyplot(loss_fig)

if __name__ == "__main__":
    main()





