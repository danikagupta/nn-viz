import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import graphviz

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        
        # Store previous epoch's weights for comparison
        self.previous_fc1_weights = self.fc1.weight.clone()
        self.previous_fc1_bias = self.fc1.bias.clone()
        self.previous_fc2_weights = self.fc2.weight.clone()
        self.previous_fc2_bias = self.fc2.bias.clone()
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc1.bias, a=-0.1, b=0.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def get_weight_changes(self, threshold=1.0):
        # Calculate percentage changes between current and previous epoch
        fc1_weight_changes = (self.fc1.weight - self.previous_fc1_weights) / (torch.abs(self.previous_fc1_weights) + 1e-8) * 100
        fc1_bias_changes = (self.fc1.bias - self.previous_fc1_bias) / (torch.abs(self.previous_fc1_bias) + 1e-8) * 100
        fc2_weight_changes = (self.fc2.weight - self.previous_fc2_weights) / (torch.abs(self.previous_fc2_weights) + 1e-8) * 100
        fc2_bias_changes = (self.fc2.bias - self.previous_fc2_bias) / (torch.abs(self.previous_fc2_bias) + 1e-8) * 100
        
        # Filter out minimal changes based on threshold
        fc1_weight_mask = torch.abs(fc1_weight_changes) >= threshold
        fc1_bias_mask = torch.abs(fc1_bias_changes) >= threshold
        fc2_weight_mask = torch.abs(fc2_weight_changes) >= threshold
        fc2_bias_mask = torch.abs(fc2_bias_changes) >= threshold
        
        # Mask out minimal changes
        fc1_weight_changes[~fc1_weight_mask] = 0
        fc1_bias_changes[~fc1_bias_mask] = 0
        fc2_weight_changes[~fc2_weight_mask] = 0
        fc2_bias_changes[~fc2_bias_mask] = 0
        
        # Update previous weights for next comparison
        self.previous_fc1_weights = self.fc1.weight.clone()
        self.previous_fc1_bias = self.fc1.bias.clone()
        self.previous_fc2_weights = self.fc2.weight.clone()
        self.previous_fc2_bias = self.fc2.bias.clone()
        
        return {
            'fc1_weight_changes': fc1_weight_changes,
            'fc1_bias_changes': fc1_bias_changes,
            'fc2_weight_changes': fc2_weight_changes,
            'fc2_bias_changes': fc2_bias_changes
        }

def visualize_architecture_with_weight_changes(model, epoch, weight_changes=None):
    dot = graphviz.Digraph()
    
    # Input layer nodes
    dot.node('X1', 'Input 1')
    dot.node('X2', 'Input 2')

    # Highlight color function
    def get_color(change_percentage):
        if change_percentage > 0:
            return 'green'  # Positive change
        elif change_percentage < 0:
            return 'red'   # Negative change
        else:
            return 'black'  # No change

    # Hidden layer nodes
    for i in range(4):
        bias = model.fc1.bias[i].detach().item()
        bias_change = weight_changes['fc1_bias_changes'][i].item() if weight_changes is not None else 0
        if abs(bias_change) >= 1:
            bias_color = get_color(bias_change)
            dot.node(f'H{i+1}', f'Hidden {i+1}\nBias: {bias:.2f}\nΔ%: {bias_change:.2f}%', fontcolor=bias_color)
        else:
            dot.node(f'H{i+1}', f'Hidden {i+1}\nBias: {bias:.2f}')
    
    # Output layer node
    output_bias = model.fc2.bias[0].detach().item()
    output_bias_change = weight_changes['fc2_bias_changes'][0].item() if weight_changes is not None else 0
    if abs(output_bias_change) >= 1:
        output_bias_color = get_color(output_bias_change)
        dot.node('Y', f'Output\nBias: {output_bias:.2f}\nΔ%: {output_bias_change:.2f}%', fontcolor=output_bias_color)
    else:
        dot.node('Y', f'Output\nBias: {output_bias:.2f}')

    # Connect input to hidden layer with weights on edges
    for i in range(2):
        for j in range(4):
            weight = model.fc1.weight[j, i].detach().item()
            if weight_changes:
                weight_change = weight_changes['fc1_weight_changes'][j, i].item()
                if abs(weight_change) >= 1:
                    weight_color = get_color(weight_change)
                    dot.edge(f'X{i+1}', f'H{j+1}', label=f'w={weight:.2f}\nΔ%: {weight_change:.2f}%', fontcolor=weight_color, color=weight_color)
                else:
                    dot.edge(f'X{i+1}', f'H{j+1}', label=f'w={weight:.2f}')
            else:
                dot.edge(f'X{i+1}', f'H{j+1}', label=f'w={weight:.2f}')

    # Connect hidden to output layer with weights on edges
    for i in range(4):
        weight = model.fc2.weight[0, i].detach().item()
        if weight_changes:
            weight_change = weight_changes['fc2_weight_changes'][0, i].item()
            if abs(weight_change) >= 1:
                weight_color = get_color(weight_change)
                dot.edge(f'H{i+1}', 'Y', label=f'w={weight:.2f}\nΔ%: {weight_change:.2f}%', fontcolor=weight_color, color=weight_color)
            else:
                dot.edge(f'H{i+1}', 'Y', label=f'w={weight:.2f}')
        else:
            dot.edge(f'H{i+1}', 'Y', label=f'w={weight:.2f}')

    return dot

def main():
    st.title("Neural Network Training with Filtered Weight Change Visualization")

    # User input for training data
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

    # Model setup
    model = SimpleNN()
    criterion = nn.MSELoss()
    learning_rate = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.01, step=0.01)  # Changed default to 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Change threshold slider
    change_threshold = st.sidebar.slider('Weight Change Threshold (%)', 1, 10, 1, step=1)

    # Training loop
    num_epochs = 10
    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute weight changes (after the first epoch)
        weight_changes = model.get_weight_changes(threshold=change_threshold) if epoch > 0 else None

        # Visualization
        st.subheader(f"Model Architecture at Epoch {epoch+1}")
        if os.system("which dot") == 0:  # Check if Graphviz is installed
            dot = visualize_architecture_with_weight_changes(model, epoch+1, weight_changes)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                dot.render(tmpfile.name, format='png')
                st.image(tmpfile.name + ".png", caption=f'Model Architecture with Weight Changes at Epoch {epoch+1}', use_container_width=True)
        else:
            st.warning("Graphviz is not installed. Please install to visualize model architecture.")

        # Display epoch details
        st.write(f"### Epoch {epoch+1}")
        st.write(f"**Loss:** {loss.item():.4f}")

        # Display weight changes (except for first epoch)
        if weight_changes:
            st.write("#### Significant Weight Changes")
            
            # FC1 Layer (Input to Hidden) Weights
            fc1_changes = weight_changes['fc1_weight_changes']
            significant_fc1_changes = torch.where((fc1_changes != 0))[0]
            if significant_fc1_changes.numel() > 0:
                st.write("**FC1 Layer (Input to Hidden) Weight Changes:**")
                for idx in significant_fc1_changes:
                    row, col = idx // 2, idx % 2
                    st.write(f"Weight (Input {col+1} → Hidden {row+1}): {fc1_changes[row, col].item():.2f}%")
            
            # FC2 Layer (Hidden to Output) Weights
            fc2_changes = weight_changes['fc2_weight_changes']
            significant_fc2_changes = torch.where((fc2_changes != 0))[0]
            if significant_fc2_changes.numel() > 0:
                st.write("**FC2 Layer (Hidden to Output) Weight Changes:**")
                for idx in significant_fc2_changes:
                    st.write(f"Weight (Hidden {idx+1} → Output): {fc2_changes[0, idx].item():.2f}%")

        # Store loss
        losses.append(loss.item())

    # Plot loss
    st.subheader("Training Loss")
    fig, ax = plt.subplots()
    ax.plot(range(len(losses)), losses, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()





