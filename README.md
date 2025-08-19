# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Jeba Solomon Raj S
### Register Number: 212223230089

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 14)
        self.fc4 = nn.Linear(14, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

Jeba=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(Jeba.parameters(), lr=0.001)

def train_model(Jeba, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(Jeba(X_train), y_train)
        loss.backward()
        optimizer.step()

        Jeba.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(Jeba, X_train_tensor, y_train_tensor, criterion, optimizer)

```
## Dataset Information

<img width="561" height="237" alt="data" src="https://github.com/user-attachments/assets/1d3fcd44-1fe6-4b29-baa8-dfb3cceee334" />

## OUTPUT

<img width="621" height="354" alt="ex-111" src="https://github.com/user-attachments/assets/81f1da54-2b0d-49f9-aab4-81bef920c9e4" />

## Training Loss Vs Iteration Plot
<img width="582" height="456" alt="Ex-1" src="https://github.com/user-attachments/assets/91321bd4-24a1-4673-a44a-b54659be2403" />


<img width="588" height="29" alt="ex-12" src="https://github.com/user-attachments/assets/7b89fa86-a412-4e16-832e-ada5cdfb0a96" />

## New Data Sample Prediction :
<img width="593" height="23" alt="pred" src="https://github.com/user-attachments/assets/3aa47dde-11fb-4fd1-9af3-70e1afaac182" />

## RESULT

The result has been successfully verified.
