import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    layout="wide",
    page_title="MLP Training Visualization",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS for authors section
st.markdown("""
    <style>
    .authors-section {
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .authors-header {
        color: #0f52ba;
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
        line-height: 1.2;
    }
    .author-name {
        font-size: 1.2rem;
        color: #262730;
        margin: 0.3rem 0;
        font-weight: 500;
        line-height: 1.2;
    }
    </style>
""", unsafe_allow_html=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Network Architecture
input_size = 28 * 28  # 784 nodes
output_size = 10      # 10 classes

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def create_network_visualization(input_size, hidden_size, output_size):
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Interactive visualization of Multi-Layer Perceptron (MLP) training on FashionMNIST dataset">
        <meta name="keywords" content="Neural Network, MLP, Deep Learning, FashionMNIST, Interactive Visualization, AI">
        <meta name="authors" content="Thi-Ngoc-Truc Nguyen, Hoang-Nguyen Vu">
        <meta property="og:title" content="Interactive MLP Training Visualization">
        <meta property="og:description" content="Visualize the training process of a Multi-Layer Perceptron on FashionMNIST dataset">
        <meta property="og:type" content="website">
        <title>Interactive MLP Training Visualization</title>
        <style>
            #network-container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div id="network-container">
            <canvas id="networkCanvas" width="1200" height="600"></canvas>
        </div>
        
        <script>
            const canvas = document.getElementById('networkCanvas');
            const ctx = canvas.getContext('2d');
            
            // Network configuration
            const layers = [{input_size}, {hidden_size}, {output_size}];
            const xSpacing = 400;
            const nodeRadius = 25;
            const colors = ['#a8d5ff', '#98ff98', '#ffcccb'];
            const borderColors = ['#4a90e2', '#50c878', '#e74c3c'];
            let animationTime = 0;
            let isForward = true;
            let transitionProgress = 0;
            
            function easeInOutSine(x) {{
                return -(Math.cos(Math.PI * x) - 1) / 2;
            }}
            
            function drawNode(x, y, color, borderColor, isActive = false) {{
                // Draw node shadow
                ctx.beginPath();
                ctx.arc(x, y + 2, nodeRadius, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.fill();
                
                // Draw node glow when active
                if (isActive) {{
                    ctx.beginPath();
                    ctx.arc(x, y, nodeRadius * 1.3, 0, Math.PI * 2);
                    const gradient = ctx.createRadialGradient(x, y, nodeRadius * 0.5, x, y, nodeRadius * 1.3);
                    gradient.addColorStop(0, color);
                    gradient.addColorStop(1, 'rgba(255,255,255,0)');
                    ctx.fillStyle = gradient;
                    ctx.fill();
                }}
                
                // Draw node
                ctx.beginPath();
                ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = borderColor;
                ctx.lineWidth = 2;
                ctx.stroke();
            }}
            
            function drawEdge(x1, y1, x2, y2, progress = 1, isBackward = false) {{
                // Draw static edge
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = 'rgba(200, 200, 200, 0.2)';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                // Draw animated signal
                if (progress > 0 && progress < 1) {{
                    let startX, startY, endX, endY;
                    if (isBackward) {{
                        startX = x2;
                        startY = y2;
                        endX = x1;
                        endY = y1;
                    }} else {{
                        startX = x1;
                        startY = y1;
                        endX = x2;
                        endY = y2;
                    }}
                    
                    const currentX = startX + (endX - startX) * progress;
                    const currentY = startY + (endY - startY) * progress;
                    
                    // Draw signal path with gradient
                    const gradient = ctx.createLinearGradient(startX, startY, currentX, currentY);
                    const signalColor = isBackward ? 'rgba(231, 76, 60,' : 'rgba(52, 152, 219,';
                    gradient.addColorStop(0, signalColor + '0.1)');
                    gradient.addColorStop(0.5, signalColor + '0.5)');
                    gradient.addColorStop(1, signalColor + '0.1)');
                    
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(currentX, currentY);
                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    // Draw signal particle with glow
                    ctx.beginPath();
                    const particleGradient = ctx.createRadialGradient(
                        currentX, currentY, 0,
                        currentX, currentY, 6
                    );
                    particleGradient.addColorStop(0, isBackward ? '#e74c3c' : '#3498db');
                    particleGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
                    ctx.fillStyle = particleGradient;
                    ctx.arc(currentX, currentY, 6, 0, Math.PI * 2);
                    ctx.fill();
                }}
            }}
            
            function drawNetwork() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                animationTime += 0.008;
                
                // Smooth direction transition
                const cycleTime = 4;
                const cycleProgress = (animationTime % cycleTime) / cycleTime;
                isForward = cycleProgress < 0.5;
                
                // Calculate signal progress
                let totalProgress;
                if (isForward) {{
                    totalProgress = (cycleProgress * 2);
                }} else {{
                    totalProgress = ((cycleProgress - 0.5) * 2);
                }}
                
                // Apply easing to make movement smooth
                totalProgress = easeInOutSine(totalProgress);
                
                // Calculate transition progress for smooth phase change
                transitionProgress = Math.abs(cycleProgress - 0.5) * 2;
                
                // Draw titles
                ctx.font = 'bold 20px Arial';
                ctx.fillStyle = '#333';
                ctx.textAlign = 'center';
                ctx.fillText('784 (28 * 28) Node', 200, 50);
                ctx.fillText('Input Layer', 200, 80);
                
                ctx.fillText('{hidden_size} Node', 600, 50);
                ctx.fillText('Hidden Layer', 600, 80);
                
                ctx.fillText('10 Node', 1000, 50);
                ctx.fillText('One-hot', 1000, 80);
                ctx.fillText('Output Layer', 1000, 110);
                
                // Draw phase indicator with smooth transition
                ctx.font = 'bold 24px Arial';
                const forwardColor = [52, 152, 219];
                const backwardColor = [231, 76, 60];
                const r = Math.round(backwardColor[0] * (1 - transitionProgress) + forwardColor[0] * transitionProgress);
                const g = Math.round(backwardColor[1] * (1 - transitionProgress) + forwardColor[1] * transitionProgress);
                const b = Math.round(backwardColor[2] * (1 - transitionProgress) + forwardColor[2] * transitionProgress);
                ctx.fillStyle = `rgb(${{r}}, ${{g}}, ${{b}})`;
                ctx.fillText(isForward ? 'Feed Forward' : 'Backpropagation', 600, 550);
                
                const startX = 200;
                const startY = 200;
                const verticalSpacing = 70;
                
                // Define visible nodes for each layer
                const visibleNodes = [6, 4, 5];
                const positions = [];
                
                // Calculate positions
                for (let i = 0; i < layers.length; i++) {{
                    const layerX = startX + i * xSpacing;
                    const nodes = [];
                    
                    for (let j = 0; j < visibleNodes[i]; j++) {{
                        const y = startY + j * verticalSpacing;
                        nodes.push({{x: layerX, y: y}});
                    }}
                    
                    if (layers[i] > visibleNodes[i]) {{
                        nodes.push({{x: layerX, y: startY + (visibleNodes[i] + 1) * verticalSpacing, isEtc: true}});
                    }}
                    
                    positions.push(nodes);
                }}
                
                // Draw edges and signals
                for (let i = 0; i < positions.length - 1; i++) {{
                    const currentLayer = positions[i];
                    const nextLayer = positions[i + 1];
                    
                    for (let j = 0; j < currentLayer.length; j++) {{
                        if (!currentLayer[j].isEtc) {{
                            for (let k = 0; k < nextLayer.length; k++) {{
                                if (!nextLayer[k].isEtc) {{
                                    drawEdge(
                                        currentLayer[j].x, currentLayer[j].y,
                                        nextLayer[k].x, nextLayer[k].y,
                                        totalProgress,
                                        !isForward
                                    );
                                }}
                            }}
                        }}
                    }}
                }}
                
                // Draw nodes with smooth pulsing
                for (let i = 0; i < positions.length; i++) {{
                    const layer = positions[i];
                    for (let j = 0; j < layer.length; j++) {{
                        if (layer[j].isEtc) {{
                            ctx.font = '24px Arial';
                            ctx.fillStyle = '#666';
                            ctx.textAlign = 'center';
                        }} else {{
                            const pulsePhase = (animationTime * 2 + j * 0.5) % (2 * Math.PI);
                            const isActive = (Math.sin(pulsePhase) + 1) / 2 > 0.3;
                            drawNode(layer[j].x, layer[j].y, colors[i], borderColors[i], isActive);
                        }}
                    }}
                }}
                
                requestAnimationFrame(drawNetwork);
            }}
            
            // Start animation
            drawNetwork();
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=700)

def display_layer_weights(layer, layer_name):
    weights = layer.weight.detach().cpu().numpy()
    rows, cols = weights.shape
    
    # Determine which rows and columns to show
    show_rows = list(range(8)) + ['...'] + list(range(max(8, rows-8), rows)) if rows > 16 else list(range(rows))
    show_cols = list(range(8)) + ['...'] + list(range(max(8, cols-8), cols)) if cols > 16 else list(range(cols))
    
    # Create the data matrix
    data = []
    for i in show_rows:
        row_data = []
        for j in show_cols:
            if i == '...' or j == '...':
                row_data.append('...')
            else:
                row_data.append(f'{weights[i,j]:.6f}')
        data.append(row_data)
    
    # Create column labels
    if cols > 16:
        col_labels = [f'In {j+1}' for j in range(8)] + ['...'] + [f'In {j+1}' for j in range(max(8, cols-8), cols)]
    else:
        col_labels = [f'In {j+1}' for j in range(cols)]
    
    # Create row labels
    if rows > 16:
        row_labels = [f'N {i+1}' for i in range(8)] + ['...'] + [f'N {i+1}' for i in range(max(8, rows-8), rows)]
    else:
        row_labels = [f'N {i+1}' for i in range(rows)]
    
    # Create DataFrame with styling
    df = pd.DataFrame(data, columns=col_labels, index=row_labels)
    return df

# Main title with description
st.title('🧠 Interactive MLP Training Visualization')
st.markdown("""
This interactive visualization demonstrates the training process of a Multi-Layer Perceptron (MLP) 
on the FashionMNIST dataset. You can adjust the network architecture and training parameters 
to see how they affect the model's performance.
""")

# Authors info with enhanced styling
st.sidebar.markdown('<div class="authors-section">', unsafe_allow_html=True)
st.sidebar.markdown('<p class="authors-header">👩‍💻 Authors</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="author-name">Thi-Ngoc-Truc Nguyen</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="author-name">Hoang-Nguyen Vu</p>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Network parameters in sidebar with descriptions
st.sidebar.markdown("### 🔧 Model Configuration")
with st.sidebar:
    st.markdown("#### Network Architecture")
    st.markdown("Configure the size of the hidden layer in the neural network.")
    hidden_size = st.slider('Hidden Layer Size', 32, 256, 128, 32,
                           help="Number of neurons in the hidden layer. A larger size may capture more complex patterns but requires more computation.")
    
    st.markdown("#### Training Parameters")
    st.markdown("Fine-tune the learning process of the neural network.")
    learning_rate = st.slider('Learning Rate', 0.001, 0.01, 0.005, 0.001,
                             help="Step size for updating model weights. Higher values may learn faster but risk overshooting optimal weights.")
    batch_size = st.slider('Batch Size', 32, 256, 64, 32,
                          help="Number of samples processed before updating model weights. Larger batches provide more stable updates but require more memory.")
    num_epochs = st.slider('Number of Epochs', 1, 20, 5, 1,
                          help="Number of complete passes through the training dataset.")

# Network Architecture visualization with description
st.header('📊 Network Architecture')
st.markdown("Visual representation of the neural network structure showing input layer (784 nodes), "
           f"hidden layer ({hidden_size} nodes), and output layer (10 nodes).")
create_network_visualization(input_size, hidden_size, output_size)

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training section
st.header('📈 Training Progress')
st.markdown("Real-time visualization of the model's training metrics.")

# Create layout for training visualization
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Loss Over Time")
    st.markdown("Shows how well the model is learning (lower is better)")
    loss_plot = st.empty()
with col2:
    st.markdown("#### Accuracy Over Time")
    st.markdown("Percentage of correct predictions on training data")
    accuracy_plot = st.empty()

# Create containers for weights visualization
st.header('🔍 Layer Weights')
st.markdown("Visualization of the learned weights in each layer of the network.")
weight_col1, weight_col2 = st.columns(2)
with weight_col1:
    st.subheader('Hidden Layer Weights')
    st.markdown(f'Matrix Shape: {model.layer1.weight.shape}')
    hidden_weights = st.empty()
with weight_col2:
    st.subheader('Output Layer Weights')
    st.markdown(f'Matrix Shape: {model.layer2.weight.shape}')
    output_weights = st.empty()

# Initialize metrics
losses = []
accuracies = []
epochs = []

# Training control
st.markdown("### Training Control")
st.markdown("Click the button below to start the training process with the selected parameters.")
if st.button('▶️ Start Training', key='train_button'):
    fig_loss = go.Figure()
    fig_accuracy = go.Figure()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = num_epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            current_step += 1
            progress = current_step / total_steps
            progress_bar.progress(progress)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 49:  # Update every 50 batches
                avg_loss = running_loss / 50
                accuracy = 100 * correct / total
                
                losses.append(avg_loss)
                accuracies.append(accuracy)
                epochs.append(epoch + i/len(train_loader))
                
                # Update loss plot
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs, 
                    y=losses, 
                    mode='lines',
                    line=dict(color='#e74c3c', width=2),
                    name='Loss'
                ))
                fig_loss.update_layout(
                    title='Training Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(t=30, b=40, l=40, r=20),
                    font=dict(size=14)
                )
                loss_plot.plotly_chart(fig_loss, use_container_width=True)
                
                # Update accuracy plot
                fig_accuracy = go.Figure()
                fig_accuracy.add_trace(go.Scatter(
                    x=epochs, 
                    y=accuracies, 
                    mode='lines',
                    line=dict(color='#2ecc71', width=2),
                    name='Accuracy'
                ))
                fig_accuracy.update_layout(
                    title='Training Accuracy',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy (%)',
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(t=30, b=40, l=40, r=20),
                    font=dict(size=14)
                )
                accuracy_plot.plotly_chart(fig_accuracy, use_container_width=True)
                
                # Update weights display
                hidden_weights.dataframe(
                    display_layer_weights(model.layer1, 'Hidden Layer'),
                    use_container_width=True
                )
                output_weights.dataframe(
                    display_layer_weights(model.layer2, 'Output Layer'),
                    use_container_width=True
                )
                
                # Update status with styled text
                status_text.markdown(f"""
                    Epoch: {epoch+1}/{num_epochs}  
                    Batch: {i+1}/{len(train_loader)}  
                    Loss: {avg_loss:.4f}  
                    Accuracy: {accuracy:.2f}%
                """)
                
                running_loss = 0.0
                correct = 0
                total = 0
        
    progress_bar.progress(1.0)
    st.success('Training completed!') 

