import torch
import pandas as pd
import numpy as np

# Carregando os dados
data = pd.read_csv('dados_banco.csv')

# Dividindo os dados em conjunto de treinamento e teste
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Normalizando os dados
train_mean, train_std = train_data.mean(), train_data.std()
train_data = (train_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std

# Transformando as variáveis categóricas em variáveis numéricas
cat_cols = ['forma_pagamento_cartao']
train_data_encoded = pd.get_dummies(train_data, columns=cat_cols)
test_data_encoded = pd.get_dummies(test_data, columns=cat_cols)

# Convertendo os dados para tensores
train_x, train_y = torch.tensor(train_data_encoded.get('transferencia_pix').values, dtype=torch.float32)

test_x, test_y = torch.tensor(test_data_encoded['transferencia_pix'].values, dtype=torch.float32)
                 
if 'transferencia_pix' not in data.columns:
    raise ValueError("Column 'transferencia_pix' not found in dataset")
if 'forma_pagamento_cartao' not in data.columns:
    raise ValueError("Column 'forma_pagamento_cartao' not found in dataset")

# Criando a classe do modelo
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                self.layers.append(torch.nn.Sigmoid())
            else:
                self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Definindo a função de perda e o otimizador
model = Model(train_x.shape[1], [10, 5], 1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Treinando o modelo
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(train_x)
    loss = criterion(y_pred, train_y.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch: {} - Loss: {:.4f}'.format(epoch, loss.item()))


# Avaliando o desempenho do modelo
with torch.no_grad():
    y_pred = model(test_x)
    predicted_classes = (y_pred > 0.5).float()
    accuracy = (predicted_classes.squeeze() == test_y).float().mean()

print('Accuracy: {:.4f}'.format(accuracy.item()))

# Fazendo previsões
new_data = pd.DataFrame({'forma_pagamento_cartao': [1, 0], 'valor': [1000, 500]})
new_data['forma_pagamento_cartao'] = new_data['forma_pagamento_cartao'].astype('category')
new_data_encoded = pd.get_dummies(new_data, columns=['forma_pagamento_cartao'])
new_data_normalized = (new_data_encoded - train_mean.drop(['transferencia_pix'])) / train_std.drop(['transferencia_pix'])

new_x = torch.tensor(new_data_normalized.values, dtype=torch.float32)

with torch.no_grad():
    y_pred = model(new_x)
    predicted_classes = (y_pred > 0.5).float()

print('Predicted class for new data: {}'.format(predicted_classes.squeeze().tolist()))

