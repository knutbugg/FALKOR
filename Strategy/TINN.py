import torch


class GRUTechNet(torch.nn.Module):

    def __init__(self, num_features, hidden_size, num_layers, output_size):
        super(GRUTechNet, self).__init__()
        self.rnn = torch.nn.GRU(num_features, hidden_size, num_layers,
              dropout=0, bidirectional=False)
        # Linear layer used to take GRU outputted hidden layer and transform
        # to buy/hold/sell signal
        self.lin = torch.nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.lin(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden


# Train model
model = torch.Model(input_size,output_size,hidden_dim,n_layers)
# send model to gpu
model.to(device)

# Define hyperparameters
n_epochs = 100
lr = 0.01

# Define loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq.to(device)
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
