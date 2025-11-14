########################## 
#NORMALIZATION


# Frequenze delle classi nel training set (dopo label encoding)
#train_counts = df_train['label_enc'].value_counts().sort_index().values

# Calcolo pesi inversamente proporzionali alla frequenza
#class_weights = 1.0 / torch.tensor(train_counts, dtype=torch.float32)

# Normalizzazione (raccomandata)
#class_weights = class_weights / class_weights.sum()


# Calculate the frequency (count) of each class in the training labels.
counts = df_train['label_enc'].value_counts().sort_index().values
# Calculate the total number of training samples.
total = counts.sum()
# Determine the total number of unique classes.
num_classes = len(counts)

# Calculate class weights using the "inverse frequency" method, normalized by the number of classes.
# Formula: class_weight_i = Total_Samples / (Count_of_Class_i * Num_Classes)
# This assigns higher weights to less frequent (minority) classes to combat class imbalance.
class_weights = total / (counts * num_classes)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class counts (train):", train_counts)
print("Class weights:", class_weights)




####################################################################################################

##################################################################################################

# ----------------------------
# MODEL
# ----------------------------
# Create model and display architecture with parameter count
rnn_model = RecurrentClassifier(
    input_size=input_shape[-1], # Pass the number of features
    hidden_size=HIDDEN_SIZE,
    num_layers=HIDDEN_LAYERS,
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE,
    bidirectional=False, #True #False
    rnn_type='GRU'  # 'RNN', 'LSTM', or 'GRU'
    ).to(device)
recurrent_summary(rnn_model, input_size=input_shape)

# Set up TensorBoard logging and save model architecture
experiment_name = "rnn"
writer = SummaryWriter("./"+logs_dir+"/"+experiment_name)
x = torch.randn(1, input_shape[0], input_shape[1]).to(device)
writer.add_graph(rnn_model, x)


# ----------------------------
# OPTIMIZER & MIXED PRECISION
# ----------------------------

# Define optimizer with L2 regularization
optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
#optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)

# Enable mixed precision training for GPU acceleration
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

# ----------------------------
# CRITERION
# ----------------------------

#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights)