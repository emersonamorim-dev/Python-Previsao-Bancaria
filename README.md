# Python-Previsao-Bancaria

## Neural Network para previsão de pagamento por transferência PIX
Este repositório contém uma implementação de uma rede neural usando PyTorch para previsão de pagamento por transferência PIX baseado em dados de histórico de pagamento.

Dependências
1 - Python 3.x
2 - PyTorch
3 - Pandas
4 - Numpy
5 - Conjunto de dados

O conjunto de dados utilizado é o dados_banco.csv que contém informações sobre as formas de pagamento utilizadas pelos clientes de um banco.

- Executando o código
Clone o repositório em sua máquina local
Instale as dependências listadas acima
Execute o arquivo main.py
Descrição do código
O código realiza as seguintes etapas:

- Carrega os dados do arquivo dados_banco.csv.
- Divide os dados em conjunto de treinamento e teste.
Normaliza os dados.
- Transforma as variáveis categóricas em variáveis numéricas.
- Converte os dados para tensores.
- Cria a classe do modelo de rede neural.
- Define a função de perda e o otimizador.
- Treina o modelo.
- Avalia o desempenho do modelo usando o conjunto de teste.
- Realiza previsões para um novo conjunto de dados.

O modelo é uma rede neural com duas camadas ocultas, com 10 e 5 neurônios respectivamente. A função de ativação utilizada é a sigmoid e a função de perda é a Binary Cross Entropy (BCELoss).

O código imprime o valor da perda a cada 100 épocas durante o treinamento e, ao final, exibe a acurácia do modelo no conjunto de teste. Por fim, realiza uma previsão para um novo conjunto de dados.
