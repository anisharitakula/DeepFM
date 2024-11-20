class BaseLearner:
    def __init__(self,train_dl,test_dl,model,optimizer,lr,criterion):
        self.train_dl=train_dl
        self.test_dl=test_dl
        self.model=model
        self.optimizer=optimizer
        self.lr=lr
        self.criterion=criterion
    
    def fit(self,epochs):
        for epoch in range(epochs):
            train_loss=self._train_epoch()
            val_loss=self._val_epoch()
            print(f"Train loss after epoch {epoch} is {train_loss}")
            print(f"Val loss after epoch {epoch} is {val_loss}")
            print("/n")
        return val_loss
    
    def _train_epoch(self):
        running_loss=0
        for i,data in enumerate(self.train_dl):
            
            input_data,label=data
            
            #Train mode
            self.model.train()
            #Make prediction
            pred=self.model(input_data)

            #print(f"The shape of pred is {pred.shape} and shape of label is {label.shape}")
            #Loss calculation and gradients calculation
            loss=self.criterion(pred,label)
            loss.backward()

            #Adjusting parameters/weights
            self.optimizer.step()

            #Gather data and report
            running_loss+=loss.item()

            #Zero out gradients
            self.optimizer.zero_grad()

        return running_loss/i
    
    def _val_epoch(self):
        running_loss=0
        for i,data in enumerate(self.test_dl):
            
            input_data,label=data
            
            #Train mode
            self.model.eval()
            #Make prediction
            pred=self.model(input_data)

            #Loss calculation 
            loss=self.criterion(pred,label)
            running_loss+=loss.item()

        return running_loss/i
