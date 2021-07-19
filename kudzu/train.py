import numpy as np

class Learner:
    
    def __init__(self, loss, model, opt, epochs):
        self.loss = loss
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.cbs = []
    def set_callbacks(self, cblist):
        for cb in cblist:
            self.cbs.append(cb)
            
    def __call__(self, cbname, *args):
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status
    
    def train_loop(self, dl, X_test_data, y_test_data):
        self.dl = dl  # dl added in here
        self.X_test_data = X_test_data
        self.y_test_data = y_test_data
        bs = self.dl.bs
        datalen = len(self.dl.data)
        self.bpe = datalen//bs
        self.afrac = 0.
        if datalen % bs > 0:
            self.bpe  += 1
            self.afrac = (datalen % bs)/bs
        self('fit_start')
        for epoch in range(self.epochs):
            self('epoch_start', epoch)
            total_predictions = []
            total_values, inputs_third = [],[]
            for inputs, targets in dl:
            
                self("batch_start", dl.current_batch)
                
                # make predictions
                predicted = self.model(inputs)
                predicted_1 = predicted.copy()
                predicted_train = predicted_1.reshape(-1,1)
                predicted_train = 1*(predicted_train >= 0.5)
                prob_train = np.array(1*(predicted_train == targets))

                # actual loss value
                epochloss = self.loss(predicted, targets)
                self('after_loss', epochloss)

                # calculate gradient
                intermed = self.loss.backward(predicted, targets)
                self.model.backward(intermed)

                # make step
                self.opt.step(self.model)
                total_predictions.append(np.sum(prob_train))
                total_values.append(prob_train.shape[0])
                
                
            
                self('batch_end')
            
           
            total_train_prob = sum(total_predictions)/sum(total_values)
            predicted_test = self.model(self.X_test_data)
            predicted_test = 1*(predicted_test>=0.5)
            prob_test_set = np.mean(1*(predicted_test == self.y_test_data))
            self('epoch_end',total_train_prob, prob_test_set, predicted_test)
        self('fit_end')
        return epochloss