import numpy as np
import pickle

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.dW=None
        self.db=None
        
        self.x = None
    
    def forward(self, x):
        self.x = x #acts as x for the first layer and AL-1 for the hidden layers
        Z = np.dot(x, self.W) + self.b
        
        return Z
    
    def backward(self, upstream_grad):
        self.dW = np.dot(self.x.T, upstream_grad)
        self.db = np.sum(upstream_grad, axis=0, keepdims=True)
        dX = np.dot(upstream_grad, self.W.T)
        
        return dX
    
    
class ReLU:
    
    def forward(self,z):
        self.z=z
        return np.maximum(0, z)
                          
    def backward(self,upstream_grad):
        derivative= (self.z >0)
        return upstream_grad * derivative
    
# class Sigmoid:
#     def __init__(self,logits,upstream_grad):
#         self.logits=logits
#         self.upstream_grad=upstream_grad
#     def forward(self):
#         return np.sigmoid(self.logits)
#     def backward(self):
#         return np.dot((np.sigmoid(self.logits)*(1-np.sigmoid(self.logits))),self.upstream_grad)
class Sigmoid:
        
    def forward(self,z):
        self.A= 1/(1+np.exp(-z))
        return self.A
    
    def backward(self,upstream_grad):
        return (self.A*(1-self.A))*upstream_grad
    
class Tanh:

    def forward(self,z):
        self.A= np.tanh(z)
        return self.A
    def backward(self,upstream_grad):
        return (1-self.A**2)*upstream_grad
    
class Softmax:
    
    def forward(self, z):
        z_max=np.max(z, axis=1, keepdims=True)
        z_clipped=z - z_max
        
        exp_logits=np.exp(z_clipped)
        self.S = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return self.S
    
    def backward(self, upstream_grad):
        grads=[]
        m, K = self.S.shape
        
        for i in range(m):
            s_i = self.S[i].reshape(1, -1)
            g_i = upstream_grad[i].reshape(1, -1)
            
            J_i = np.diagflat(s_i) - np.dot(s_i.T, s_i)
            
            grad_i = np.dot(g_i, J_i)
            grads.append(grad_i)
            
        return np.concatenate(grads, axis=0)

class CrossEntropyLoss:
    
    def forward(self, logits, y_true):
        clipped_logits= logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(clipped_logits)
        self.S = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.y_true = y_true
        self.m = logits.shape[0]
        
        loss = -np.sum(y_true * np.log(self.S + 1e-9)) / self.m
        return loss
    
    def backward(self):
        return (self.S - self.y_true) / self.m
    
class MSELoss:
    
    def forward(self, predictions, y_true):
        self.predictions=predictions
        self.y_true=y_true
        
        diff=self.predictions-self.y_true
        m=self.predictions.shape[0]
        
        loss = np.sum(diff ** 2) / m
        
        return loss 
    
    def backward(self):
        
        m = self.predictions.shape[0]
        
        return 2 * (self.predictions - self.y_true) / m
    
class SGD:
    def __init__(self, lr = 0.01):
        self.lr=lr
        
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer.W -= self.lr * layer.dW
            
            if hasattr(layer, 'b'):
                layer.b -= self.lr * layer.db
            
    
class Model:
    def __init__(self):
        self.layers=[]
        self.loss_fn=None
        self.optimizer=None
        
    def add_layer(self, layer):
        
        self.layers.append(layer)
        
    def compile(self, loss_fn, optimizer):
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        
    def train(self, x_train, y_train, epochs, batch_size, shuffle=True):
        
        for epoch in range(epochs):
            
            if shuffle:
                sample_idxs=np.arange(x_train.shape[0])
                np.random.shuffle(sample_idxs)
                x_train=x_train[sample_idxs]
                y_train=y_train[sample_idxs]
                
            epoch_loss=0
            num_batches=0
            
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
                    
                loss = self.loss_fn.forward(output, y_batch)
                epoch_loss += loss
                num_batches += 1
                
                grad = self.loss_fn.backward()
                
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                    
                self.optimizer.step(self.layers)
                
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                
                  
    def predict(self, input_data):
        output=input_data
        for layer in self.layers:
            output=layer.forward(output)
        return output
        
    def evaluate(self, x_test, y_test):
        predictions=self.predict(x_test)
        loss=self.loss_fn.forward(predictions, y_test)
        
        pred_labels=np.argmax(predictions, axis=1)
        true_labels=np.argmax(y_test, axis = 1)
        accuracy= np.mean(pred_labels==true_labels)
        
        print(f"Evaluation -> Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
        return loss, accuracy
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
        print(f"Model saved to {filename}")
    def load(self, filename):
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
            
        self.layers = loaded_model.layers
        self.loss_fn = loaded_model.loss_fn
        self.optimizer = loaded_model.optimizer
        print(f"Model loaded from {filename}")

