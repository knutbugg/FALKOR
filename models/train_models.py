
class TrainModel:
	"""Simple training class for a PyTorch model"""

	def __init__():
		pass

	def train(self, model, num_epochs, batch_size, train_gen, valid_gen, test_gen, gru=False):
	    """Standard training function used by all three models"""
	    
	    # For optimizing our model, we choose SGD 
	    optimizer = optim.Adam(model.parameters(), lr=1e-1)
	    
	    # training loop
	    
	    # toop through the dataset num_epoch times
	    for epoch in range(num_epochs):
	               
	        # train loop
	        train_loss = []
	        valid_loss = []
	        
	        # take the batch and labels for batch 
	        for batch, labels in train_gen:
	            
	            if gru:
	                # add extra dimension to every vector in batch
	                batch.unsqueeze_(-1)
	                batch = batch.expand(batch.shape[0], batch.shape[1], 1)
	                
	                # reformat dimensions
	                batch = batch.transpose(2,0)
	                batch = batch.transpose(1, 2)
	                
	            batch, labels = batch.cuda(), labels.cuda()
	            batch, labels = batch.float(), labels.float()
	            
	            # clear gradients
	            model.zero_grad()
	            output = model(batch)
	            
	            if gru:
	                output = output[0] # turn (1, batch_size, 1) to (batch_size, 1)
	            
	            # declare the loss function and calculate output loss
	            
	            # we use the RMSE error function to train our model
	            criterion = nn.MSELoss()
	            
	            loss = torch.sqrt(criterion(output, labels))
	            
	            # backpropogate loss through model
	            loss.backward()

	            # perform model training based on propogated loss
	            optimizer.step()
	            
	            train_loss.append(loss)
	        
	        # validation loop
	        
	        profit = 0
	        with torch.set_grad_enabled(False):
	            for batch, labels in valid_gen:
	                if gru:
	                    # add extra dimension to every vector in batch
	                    batch.unsqueeze_(-1)
	                    batch = batch.expand(batch.shape[0], batch.shape[1], 1)

	                    # reformat dimensions
	                    batch = batch.transpose(2,0)
	                    batch = batch.transpose(1, 2)
	                    
	                batch, labels = batch.cuda(), labels.cuda()
	                batch, labels = batch.float(), labels.float()
	                
	                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
	                model.eval()

	                output = model(batch)
	                
	                if gru:
	                    output = output[0] # turn (1, batch_size, 1) to (batch_size, 1)
	                
	                val_loss = torch.sqrt(criterion(output, labels))
	                
	                model.train()
	                
	                valid_loss.append(val_loss)
	                
	            
	            # Profitability testing
	            profit = 0.0
	            
	            for batch, labels in test_gen:
	                if gru:
	                    # add extra dimension to every vector in batch
	                    batch.unsqueeze_(-1)
	                    batch = batch.expand(batch.shape[0], batch.shape[1], 1)

	                    # reformat dimensions
	                    batch = batch.transpose(2,0)
	                    batch = batch.transpose(1, 2)
	                
	                batch, labels = batch.cuda(), labels.cuda()
	                batch, labels = batch.float(), labels.float()
	                
	                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
	                model.eval()
	                
	                output = model(batch)
	                if gru:
	                    output = output[0] # turn (1, batch_size, 1) to (batch_size, 1)
	                
	                # if output is > 0 ==> model predict positive growth for the next five cycles. Purchase now and sell in 5 periods.
	                for i, pred in enumerate(output):
	                    #print(pred)
	                    if pred[0] > 0: # price will increase
	                        profit += labels[i]
	                       
	                model.train()

	def train_dual(self, model, num_epochs, batch_size, train_gen1, train_gen2, valid_gen1, valid_gen2, test_gen, gru=False):
	    """Standard training function used by all three models"""
	    # For optimizing our model, we choose SGD 
	    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
	    
	    # training loop
	    
	    # toop through the dataset num_epoch times
	    for epoch in range(num_epochs):
	        
	        # train loop
	        
	        train_loss = []
	        valid_loss = []
	        
	        # loop through each batch
	        for i  in range(batch_size):
	            gru_batch, gru_labels = next(iter(train_gen1))
	            gru_batch, gru_labels = gru_batch.cuda(), gru_labels.cuda()
	            gru_batch, gru_labels = gru_batch.float(), gru_labels.float()
	            
	            # add extra dimension to every vector in batch
	            gru_batch.unsqueeze_(-1)
	            gru_batch = gru_batch.expand(gru_batch.shape[0], gru_batch.shape[1], 1)

	            # reformat dimensions
	            gru_batch = gru_batch.transpose(2,0)
	            gru_batch = gru_batch.transpose(1, 2)
	            cnn_batch, cnn_labels = next(iter(train_gen2))
	            cnn_batch, cnn_labels = cnn_batch.cuda(), cnn_labels.cuda()
	            cnn_batch, cnn_labels = cnn_batch.float(), cnn_labels.float()
	            
	            # clear gradients
	            model.zero_grad()
	            output = model(gru_batch, cnn_batch)
	            output = output[0]
	            # declare the loss function and calculate output loss
	            
	            # we use the RMSE error function to train our model
	            criterion = nn.MSELoss()
	            
	            loss = torch.sqrt(criterion(output, gru_labels))
	            
	            # backpropogate loss through model
	            loss.backward()
	            # perform model training based on propogated loss
	            optimizer.step()
	            
	            train_loss.append(loss)
	        
	        
	        # validation loop
	        with torch.set_grad_enabled(False):
	            for i in range(batch_size):
	                gru_batch, gru_labels = next(iter(valid_gen1))
	                gru_batch, gru_labels = gru_batch.cuda(), gru_labels.cuda()
	                gru_batch, gru_labels = gru_batch.float(), gru_labels.float()
	                
	                # add extra dimension to every vector in batch
	                gru_batch.unsqueeze_(-1)
	                gru_batch = gru_batch.expand(gru_batch.shape[0], gru_batch.shape[1], 1)

	                # reformat dimensions
	                gru_batch = gru_batch.transpose(2,0)
	                gru_batch = gru_batch.transpose(1, 2)

	                cnn_batch, cnn_labels = next(iter(valid_gen2))
	                cnn_batch, cnn_labels = cnn_batch.cuda(), cnn_labels.cuda()
	                cnn_batch, cnn_labels = cnn_batch.float(), cnn_labels.float()
	                
	                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
	                model.eval()

	                output = model(gru_batch, cnn_batch)
	                output = output[0]
	                
	                val_loss = torch.sqrt(criterion(output, gru_labels))
	                
	                model.train()
	                
	                valid_loss.append(val_loss)
	                
	             # Profitability testing
	            profit = 0.0
	            
	            for batch, labels in test_gen:
	                gru_batch, gru_labels = next(iter(valid_gen1))
	                gru_batch, gru_labels = gru_batch.cuda(), gru_labels.cuda()
	                gru_batch, gru_labels = gru_batch.float(), gru_labels.float()
	                
	                # add extra dimension to every vector in batch
	                gru_batch.unsqueeze_(-1)
	                gru_batch = gru_batch.expand(gru_batch.shape[0], gru_batch.shape[1], 1)

	                # reformat dimensions
	                gru_batch = gru_batch.transpose(2,0)
	                gru_batch = gru_batch.transpose(1, 2)

	                cnn_batch, cnn_labels = next(iter(valid_gen2))
	                cnn_batch, cnn_labels = cnn_batch.cuda(), cnn_labels.cuda()
	                cnn_batch, cnn_labels = cnn_batch.float(), cnn_labels.float()
	                
	                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
	                model.eval()

	                output = model(gru_batch, cnn_batch)
	                output = output[0]
	                
	                # if output is > 0 ==> model predict positive growth for the next five cycles. Purchase now and sell in 5 periods.
	                
	                for i, pred in enumerate(output):
	                    if pred > 0: # price will increase
	                        profit += labels[i]
	                
	                
	                model.train()
	                
	        print("Epoch: {}/{}...".format(epoch+1, num_epochs),
	              "Training Loss: {}".format(round(float(sum(train_loss)/len(train_loss)), 4)),
	              "Validation Loss: {}".format(round(float(sum(valid_loss)/len(valid_loss)), 4)),
	              "Profitability: {}".format(round(float(profit), 3)))            