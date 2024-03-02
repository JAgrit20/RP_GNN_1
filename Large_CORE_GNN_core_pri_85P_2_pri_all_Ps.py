#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Firefox_dataset.csv')

# Check if 'priority' column exists
if 'priority' in df.columns:
    # Create an empty DataFrame for the sampled data
    sampled_df = pd.DataFrame()

    # Get unique priorities
    unique_priorities = df['priority'].unique()

    for priority in unique_priorities:
        # Filter the DataFrame by priority
        df_priority = df[df['priority'] == priority]
        
        # Sample rows for the current priority
        if len(df_priority) >= 1000:
            sampled_rows = df_priority.sample(n=1000, random_state=1)  # Ensuring reproducibility with random_state
        else:
            # If less than 1000 rows, take all
            sampled_rows = df_priority
        
        # Append the sampled rows to the sampled_df DataFrame
        sampled_df = pd.concat([sampled_df, sampled_rows], ignore_index=True)
else:
    print("The dataset does not contain a 'priority' column.")
    
# Save the sampled DataFrame to a new CSV file if needed
sampled_df.to_csv('firefox_Pri_sampled_dataset_1k.csv', index=False)

# Output the shape of the sampled dataset to confirm
sampled_df.shape


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# # Load the dataset
file_path = "core_summary_threshold_records.csv"
# firefox_issues_df = balanced_df
firefox_issues_df = pd.read_csv(file_path)
# firefox_issues_df =firefox_issues_df 

# Assuming firefox_issues_df is already loaded and ready for use
# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a bar plot for severity value counts in the firefox_issues_df DataFrame
plt.figure(figsize=(8, 6))
sns.countplot(x='priority', data=firefox_issues_df)

plt.title('Distribution of Priority Values in Core Dataset')
plt.xlabel('Priotiy')
plt.ylabel('Count')
# plt.xticks([0, 1], ['1 (High)', '2 ','3','4','5'])  # Adjust the labels as per your severity values
plt.show()


# In[3]:


import pandas as pd

# Load the dataset
df = pd.read_csv('core_summary_threshold_records.csv')

# Check if 'priority' column exists
if 'priority' in df.columns:
    # Define a mapping from old values to new ones
    priority_mapping = {
        'P1': 1,
        'P2': 2,
        'P3': 3,
        'P4': 4,
        'P5':5
        # Add more mappings as needed
    }

    # Replace the priority values based on the mapping
    df['priority'] = df['priority'].replace(priority_mapping)
else:
    print("The dataset does not contain a 'priority' column.")

df['priority'] = pd.to_numeric(df['priority'], errors='coerce')

# Handle NaN values in 'priority' column. You can either drop them or fill them.
# Here, we'll drop any rows with NaN in 'priority'. Alternatively, you can fill with a value, e.g., df['priority'].fillna(0, inplace=True)
df.dropna(subset=['priority'], inplace=True)
print(df['priority'].value_counts)


# In[4]:


import pandas as pd

# Assuming 'name_file' is a variable containing your file path, if not replace 'name_file' with the actual file path or filename
# name_file = name_file  # Update this to your actual file path

# Load the dataset
# df = pd.read_csv(name_file)

# Check if 'severity' column exists
if 'severity' in df.columns:
    # Define a mapping from old values to new ones
    priority_mapping = {
        'S3': 1,
        'S4': 2,
        'normal': 3,
        'critical':4
        # Add more mappings as needed
    }

    # Replace the severity values based on the mapping
    df['severity'] = df['severity'].map(priority_mapping)

    # Now, we only want to keep rows where 'severity' is not NaN after mapping
    # This effectively removes rows with values not in our mapping
    df = df.dropna(subset=['severity'])

    # Optionally, convert 'severity' to an integer type if all mappings are integers
    df['severity'] = df['severity'].astype(int)
else:
    print("The dataset does not contain a 'severity' column.")

# If you want to see the value counts for the 'severity' column
print(df['severity'].value_counts())


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# # Load the dataset
# file_path = "firefox_Pri_sampled_dataset_1k.csv"
# firefox_issues_df = balanced_df
# firefox_issues_df
# firefox_issues_df =firefox_issues_df 

# Assuming firefox_issues_df is already loaded and ready for use
# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a bar plot for severity value counts in the firefox_issues_df DataFrame
plt.figure(figsize=(8, 6))
sns.countplot(x='severity', data=firefox_issues_df)

plt.title('Distribution of Priority Values in Core Dataset')
plt.xlabel('Priotiy')
plt.ylabel('Count')
# plt.xticks([0, 1], ['1 (High)', '2 ','3','4','5'])  # Adjust the labels as per your severity values
plt.show()


# In[6]:


df.head()


# In[7]:


import pandas as pd
import re
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
# # # Load the dataset
# file_path = "Core_Pri_sampled_dataset.csv"
# # firefox_issues_df = balanced_df
# firefox_issues_df = pd.read_csv(file_path)
firefox_issues_df =df 

# Clean and preprocess data
firefox_issues_df.dropna(subset=['severity', 'product', 'component', 'priority', 'status'], inplace=True)
firefox_issues_df['summary'] = firefox_issues_df['summary'].str.lower()
firefox_issues_df['summary'] = firefox_issues_df['summary'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))

firefox_issues_df.replace('--', pd.NA, inplace=True)
firefox_issues_df.dropna(subset=['severity', 'priority'], inplace=True)

# Encode 'Product', 'Component', and 'Status' using label encoding
product_encoder = LabelEncoder()
component_encoder = LabelEncoder()
status_encoder = LabelEncoder()
# firefox_issues_df['product_encoded'] = product_encoder.fit_transform(firefox_issues_df['product'])
firefox_issues_df['component_encoded'] = component_encoder.fit_transform(firefox_issues_df['component'])
firefox_issues_df['status_encoded'] = status_encoder.fit_transform(firefox_issues_df['status'])
# Encode 'severity' using label encoding
severity_encoder = LabelEncoder()
firefox_issues_df['severity_encoded'] = severity_encoder.fit_transform(firefox_issues_df['severity'])

# Convert 'keywords' list to a string
firefox_issues_df['keywords_str'] = firefox_issues_df['keywords'].apply(lambda x: ' '.join(ast.literal_eval(x)) if pd.notnull(x) else '')

# Use CountVectorizer to encode 'keywords' (This step could be memory-intensive for large datasets)
vectorizer = CountVectorizer()
keywords_encoded = vectorizer.fit_transform(firefox_issues_df['keywords_str'])

# Initialize BERT tokenizer and model with a specified cache directory
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
model = BertModel.from_pretrained('bert-base-uncased', cache_dir='cache')


# from transformers import RobertaTokenizer, RobertaModel

# # Initialize RoBERTa tokenizer and model
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base' ,cache_dir='cache')
# model = RobertaModel.from_pretrained('roberta-base', cache_dir='cache')


# Set the device to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Function for batch processing of summaries to obtain BERT embeddings
def batch_encode_summaries(summaries, tokenizer, model, batch_size=16):
    dataloader = DataLoader(summaries, batch_size=batch_size, shuffle=False)
    text_features_list = []

    for batch_summaries in dataloader:
        inputs = tokenizer(batch_summaries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_text_features = outputs.last_hidden_state.mean(dim=1)
        text_features_list.append(batch_text_features.cpu())  # Move to CPU to avoid GPU memory overload

    return torch.cat(text_features_list, dim=0)


# Encode summaries in batches to get text features
text_features = batch_encode_summaries(firefox_issues_df['summary'].tolist(), tokenizer, model, batch_size=16)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Include 'severity_encoded' in the features to scale
features_to_scale = firefox_issues_df[[ 'component_encoded', 'status_encoded', 'severity_encoded']]

# Fit the scaler to the features and transform them to a 0-1 range
scaled_features = scaler.fit_transform(features_to_scale)

# Update the dataframe with the scaled features
# firefox_issues_df[['product_encoded', 'component_encoded', 'status_encoded']] = scaled_features
# Corrected line to include 'severity_encoded' in the DataFrame update
firefox_issues_df[[ 'component_encoded', 'status_encoded', 'severity_encoded']] = scaled_features


# Convert the 'keywords_encoded' sparse matrix to a tensor
keywords_tensor = torch.tensor(keywords_encoded.toarray(), dtype=torch.float)

# Calculate 'count of blocks' and 'count of depends_on'
firefox_issues_df['blocks_count'] = firefox_issues_df['blocks'].apply(lambda x: len(ast.literal_eval(x)) if pd.notnull(x) else 0)
firefox_issues_df['depends_on_count'] = firefox_issues_df['depends_on'].apply(lambda x: len(ast.literal_eval(x)) if pd.notnull(x) else 0)

# Scale the counts
count_features = firefox_issues_df[['blocks_count', 'depends_on_count']]
scaled_count_features = scaler.fit_transform(count_features)
firefox_issues_df[['blocks_count', 'depends_on_count']] = scaled_count_features

# Prepare the features tensor, now including 'severity_encoded' and 'keywords_tensor'
features_tensor = torch.tensor(firefox_issues_df[[ 'component_encoded', 'status_encoded', 'severity_encoded', 'blocks_count', 'depends_on_count']].values, dtype=torch.float)

# Concatenate the BERT embeddings with the scaled features
features = torch.cat((text_features, features_tensor), dim=1)

# # Prepare edge index and map issue IDs to node indices
# node_id_mapping = {node_id: idx for idx, node_id in enumerate(firefox_issues_df['id'])}
# edge_index = []


# Compute cosine similarity
similarity_matrix = cosine_similarity(text_features.numpy())

# Define a similarity threshold
similarity_threshold = 0.95

# Prepare edge index
edge_index = []
node_id_mapping = {node_id: idx for idx, node_id in enumerate(firefox_issues_df['id'])}

for i in range(similarity_matrix.shape[0]):
    for j in range(i + 1, similarity_matrix.shape[1]):
        if similarity_matrix[i, j] >= similarity_threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])  # Adding reverse edge for undirected graph

edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Convert the 'priority' and 'severity' columns to tensors
# priority_labels = torch.tensor(firefox_issues_df['priority'].values, dtype=torch.long)
severity_labels = torch.tensor(firefox_issues_df['priority'].values, dtype=torch.long)

# Create PyTorch Geometric Data object
data = Data(x=features, edge_index=edge_index_tensor, y_severity=severity_labels)

# Save the Data object
data_save_path = 'firefox_issues_data.pt'
torch.save(data, data_save_path)


# In[8]:


print(firefox_issues_df['priority'].unique())


# In[9]:


# Number of Nodes
num_nodes = data.num_nodes
print(f"Number of nodes: {num_nodes}")

# Number of Edges
num_edges = data.num_edges
print(f"Number of edges: {num_edges}")

# Average Node Degree
avg_degree = num_edges / num_nodes
print(f"Average node degree: {avg_degree:.2f}")

# Number of Features per Node
num_features = data.num_features
print(f"Number of features per node: {num_features}")

# # Checking for Isolated Nodes
# num_isolated_nodes = sum(data.degree() == 0).item()
# print(f"Number of isolated nodes: {num_isolated_nodes}")

# # Checking for Self-loops
# num_self_loops = data.contains_self_loops().item()
# print(f"Number of self-loops: {num_self_loops}")

# # Graph Density
# density = num_edges / (num_nodes * (num_nodes - 1))
# print(f"Graph density: {density:.6f}")



# In[10]:


# Assuming you have defined num_classes_priority and num_classes_severity
# num_classes_priority = 6  # Example: 6 priority classes
num_classes_severity = 6  # Example: 6 severity classes

# Check the range for priority labels
# priority_label_min = data.y_priority.min().item()
# priority_label_max = data.y_priority.max().item()

# if priority_label_min < 0 or priority_label_max >= num_classes_priority:
#     print(f"Priority labels out of expected range [0, {num_classes_priority-1}]: Min = {priority_label_min}, Max = {priority_label_max}")
# else:
    # print(f"Priority labels within expected range [0, {num_classes_priority-1}].")

# Check the range for severity labels
severity_label_min = data.y_severity.min().item()
severity_label_max = data.y_severity.max().item()

if severity_label_min < 0 or severity_label_max >= num_classes_severity:
    print(f"Severity labels out of expected range [0, {num_classes_severity-1}]: Min = {severity_label_min}, Max = {severity_label_max}")
else:
    print(f"Severity labels within expected range [0, {num_classes_severity-1}].")


# In[11]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FirefoxIssueGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_units):
        super(FirefoxIssueGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_units)
        self.dropout = torch.nn.Dropout(0.5)  # Adjust dropout rate as needed
        self.out_priority = torch.nn.Linear(hidden_units, 7)
        self.out_severity = torch.nn.Linear(hidden_units, 6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        priority = self.out_priority(x)
        severity = self.out_severity(x)
        return F.log_softmax(priority, dim=1), F.log_softmax(severity, dim=1)

model = FirefoxIssueGraphSAGE(num_node_features=features.shape[1], hidden_units=110)


# In[12]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class HybridGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes_severity, num_units):
        super(HybridGNN, self).__init__()

        # First layer is a GraphSAGE layer
        self.conv1 = SAGEConv(num_node_features, num_units)

        # Second layer is a GAT layer with multi-head attention
        self.conv2 = GATConv(num_units, num_units // 2, heads=2, concat=True)

        # Final output features adjusted for concatenated multi-head attention
        final_out_features = num_units  # Assuming concat=True doubles the feature size

        # Define separate layers for 'priority' and 'severity' with adjusted class counts
        # self.out_priority = torch.nn.Linear(final_out_features, 7)  # Adjusted for 'priority'
        self.out_severity = torch.nn.Linear(final_out_features, 6)  # Correct for 'severity'

    def forward(self, x, edge_index):
        # GraphSAGE convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # Apply dropout after GraphSAGE

        # GAT convolution
        x = self.conv2(x, edge_index)
        x = F.elu(x)  # ELU activation for GAT
        x = F.dropout(x, training=self.training)  # Apply dropout after GAT

        # Output layers for 'priority' and 'severity'
        # priority = self.out_priority(x)
        severity = self.out_severity(x)

        return F.log_softmax(severity, dim=1)

# Instantiate the model with the same number of units
# model = HybridGNN(num_node_features=770, num_classes_priority=7, num_classes_severity=6, num_units=110)
model = HybridGNN(num_node_features=773, num_classes_severity=6, num_units=128)



# In[13]:


import torch
import numpy as np

# Assuming 'data' is your PyTorch Geometric Data object

# Calculate the total number of nodes
num_nodes = data.x.size(0)

# Define the split sizes
train_size = 0.70  # 70% of the data for training
val_size = 0.15  # 15% of the data for validation
test_size = 0.15  # 15% of the data for testing

# Generate shuffled indices
indices = torch.randperm(num_nodes)

# Calculate the number of nodes for each split
num_train_nodes = int(train_size * num_nodes)
num_val_nodes = int(val_size * num_nodes)

# Split the indices for each set
train_indices = indices[:num_train_nodes]
val_indices = indices[num_train_nodes:num_train_nodes + num_val_nodes]
test_indices = indices[num_train_nodes + num_val_nodes:]

# Create boolean masks
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[train_indices] = True
data.val_mask[val_indices] = True
data.test_mask[test_indices] = True

# Calculate the number of nodes in each set
num_nodes_train = data.train_mask.sum().item()
num_nodes_val = data.val_mask.sum().item()
num_nodes_test = data.test_mask.sum().item()

print(f"Number of nodes in the training set: {num_nodes_train}")
print(f"Number of nodes in the validation set: {num_nodes_val}")
print(f"Number of nodes in the test set: {num_nodes_test}")



# In[ ]:





# In[14]:


criterion_priority = torch.nn.CrossEntropyLoss()
criterion_severity = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


# In[15]:


num_epochs = 200  # Define the number of epochs
log_interval = 10  # Interval at which to log training status


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    severity_logits = model(data.x, data.edge_index)

    # Apply the mask to the entire node-wise prediction tensor
    # masked_priority_logits = priority_logits[data.train_mask]
    masked_severity_logits = severity_logits[data.train_mask]

    # Get the corresponding labels for the masked nodes
    # masked_y_priority = data.y_priority[data.train_mask.nonzero(as_tuple=True)[0]]
    masked_y_severity = data.y_severity[data.train_mask.nonzero(as_tuple=True)[0]]

    # Compute loss for each task
    # loss_priority = criterion_priority(masked_priority_logits, masked_y_priority)
    loss_severity = criterion_severity(masked_severity_logits, masked_y_severity)

    # Combine losses and perform backpropagation
    loss = loss_severity  # You might want to weight these losses differently
    loss.backward()
    optimizer.step()

    # Log training information
    if epoch % log_interval == 0:
        print(f'Epoch: {epoch}/{num_epochs}, Severity Loss: {loss_severity.item():.4f}')


# In[16]:


model.eval()
with torch.no_grad():
    # Forward pass using the entire graph
    severity_logits = model(data.x, data.edge_index)

    # Apply the test mask to logits and labels for loss calculation
    # test_priority_logits = priority_logits[data.val_mask]
    test_severity_logits = severity_logits[data.val_mask]

    # # Get the corresponding labels for the test mask
    # test_y_priority = data.y_priority[data.val_mask.nonzero(as_tuple=True)[0]]
    test_y_severity = data.y_severity[data.val_mask.nonzero(as_tuple=True)[0]]

    # Compute loss for each task
    # test_loss_priority = criterion_priority(test_priority_logits, test_y_priority)
    test_loss_severity = criterion_severity(test_severity_logits, test_y_severity)
    test_loss = test_loss_severity

    # print("prop loss", test_loss_priority.item())
    print("test_loss_severity loss", test_loss_severity.item())
    print(f'Test Loss: {test_loss.item()}')


# In[ ]:





# In[17]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model.eval()
with torch.no_grad():
    # Forward pass using the entire graph
    severity_logits = model(data.x, data.edge_index)

    test_severity_logits = severity_logits[data.test_mask]

    test_y_severity = data.y_severity[data.test_mask.nonzero(as_tuple=True)[0]]

    _, predicted_severities = torch.max(test_severity_logits, 1)

    true_severities = test_y_severity.cpu().numpy()
    predicted_severities = predicted_severities.cpu().numpy()

    # Calculate metrics for 'severity'
    accuracy_severity = accuracy_score(true_severities, predicted_severities)
    precision_severity = precision_score(true_severities, predicted_severities, average='weighted')
    recall_severity = recall_score(true_severities, predicted_severities, average='weighted')
    f1_severity = f1_score(true_severities, predicted_severities, average='weighted')

    print(f"Severity - Accuracy: {accuracy_severity:.4f}, Precision: {precision_severity:.4f}, "
          f"Recall: {recall_severity:.4f}, F1: {f1_severity:.4f}")
    report_severity = classification_report(true_severities, predicted_severities, target_names=['1','2','3','4','5'], output_dict=False)
    print("Classification Report for Severity:\n", report_severity)



# In[22]:


# Save the trained model
torch.save(model.state_dict(), 'Core_large_sum_model.pth')

# # Save the encoders and scalers
# import joblib
# # joblib.dump(product_encoder, 'C-product_encoder.pkl')
# joblib.dump(component_encoder, 'c_component_encoder.pkl')
# joblib.dump(status_encoder, 'c_status_encoder.pkl')
# joblib.dump(scaler, 'c_scaler.pkl')


# In[19]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class HybridGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes_severity, num_units):
        super(HybridGNN, self).__init__()

        # First layer is a GraphSAGE layer
        self.conv1 = SAGEConv(num_node_features, num_units)

        # Second layer is a GAT layer with multi-head attention
        self.conv2 = GATConv(num_units, num_units // 2, heads=2, concat=True)

        # Final output features adjusted for concatenated multi-head attention
        final_out_features = num_units  # Assuming concat=True doubles the feature size

        # Define separate layers for 'priority' and 'severity' with adjusted class counts
        # self.out_priority = torch.nn.Linear(final_out_features, 7)  # Adjusted for 'priority'
        self.out_severity = torch.nn.Linear(final_out_features, 6)  # Correct for 'severity'

    def forward(self, x, edge_index):
        # GraphSAGE convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # Apply dropout after GraphSAGE

        # GAT convolution
        x = self.conv2(x, edge_index)
        x = F.elu(x)  # ELU activation for GAT
        x = F.dropout(x, training=self.training)  # Apply dropout after GAT

        # Output layers for 'priority' and 'severity'
        # priority = self.out_priority(x)
        severity = self.out_severity(x)

        return F.log_softmax(severity, dim=1)

# Instantiate the model with the same number of units
# model = HybridGNN(num_node_features=770, num_classes_priority=7, num_classes_severity=6, num_units=110)
# model = HybridGNN(num_node_features=773, num_classes_severity=4, num_units=128)


# In[ ]:





# In[20]:


import torch
from sklearn.model_selection import KFold
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming HybridGNN and other necessary imports and initializations are already done

# Number of splits for K-Fold
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

# Convert labels to numpy for easier indexing
labels = data.y_severity.cpu().numpy()

results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(labels)):
    print(f'Fold {fold + 1}/{k_folds}')

    # Update masks according to the current fold
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[test_idx] = True

    # Reinitialize model and optimizer for each fold
    model = HybridGNN(num_node_features=773, num_classes_severity=4, num_units=128)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    # Training phase
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass (entire graph is used but loss is computed only for training nodes)
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out[train_mask], data.y_severity[train_mask].to(device))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Validation phase with detailed metrics
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = torch.max(out[val_mask], dim=1)[1]
        y_true = data.y_severity[val_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    })

    print(f'Fold {fold + 1}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}')

# Aggregate and print average results
avg_results = {
    'accuracy': np.mean([result['accuracy'] for result in results]),
    'precision': np.mean([result['precision'] for result in results]),
    'recall': np.mean([result['recall'] for result in results]),
    'f1': np.mean([result['f1'] for result in results]),
}

print(f"Average Results - Accuracy: {avg_results['accuracy']}, Precision: {avg_results['precision']}, Recall: {avg_results['recall']}, F1: {avg_results['f1']}")

# Visualization
sns.set_style("whitegrid")

# Plotting each metric
metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='fold', y=metric, data=pd.DataFrame(results))
    plt.title(f'{metric.title()} Across Folds')
    plt.xlabel('Fold')
    plt.ylabel(metric.title())
    plt.ylim(0, 1)
    plt.show()

