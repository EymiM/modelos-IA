import numpy as np

# representação binária dos dígitos de 0 a 9
digitos = [
    [-1,+1,+1,-1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1,-1,+1,+1,-1],
    [-1,+1,-1,-1,+1,+1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1,-1],
    [-1,+1,+1,-1,+1,-1,-1,+1,-1,-1,+1,-1,-1,+1,-1,-1,+1,+1,+1,+1],
    [+1,+1,+1,-1,-1,-1,-1,+1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1],
    [+1,-1,+1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,-1,-1,-1,+1,-1],
    [+1,+1,+1,+1,+1,-1,-1,-1,+1,+1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1],
    [-1,+1,+1,+1,+1,-1,-1,-1,+1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1],
    [+1,+1,+1,+1,-1,-1,-1,+1,-1,-1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1],
    [-1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1],
    [-1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,+1,-1,-1,-1,+1,+1,+1,+1,+1]
]

w = np.zeros((10, 20))  # pesos e bias inicializados em 0
b = np.zeros(10)
t = np.identity(10)     # targets ativados apenas quando a entrada
t = t * 2 - 1           # for seu dígito correspondente

with open('relatorio.txt', 'w') as file:
    import sys
    sys.stdout = file
    
    for j in range(10):
        i = 0
        print(f"\n\nNeuronio {j}")
        print(f"Para o neuronio {j}, apenas o digito {j} deve apresentar resultado positivo, ou seja, apenas a entrada do digito {j} deve ativar o neuronio.")
        while i < len(digitos):
            digito_atual = np.array(digitos[i])
            resultado = np.dot(w[j], digito_atual) + b[j]

            if resultado >= 0:
                saida = 1
            else:
                saida = -1

            # Compara ao target
            if saida != t[j][i % 10]:
                print(f"Para a entrada {i}, o resultado obtido foi {resultado}, ou seja a saida se torna {saida}, o que nao corresponde ao target")
                print(f"Tendo em vista que o target nao foi atingido para a entrada {i}, deve-se refazer as contas da matriz de pesos de forma a que, na proxima verificacao, nao ocorram mais erros envolvendo esta entrada.")

                w[j] += digito_atual * (t[j][i % 10] - saida)   # os pesos são recalculados
                                                                # com base na fórmula dada do
                                                                # Perceptron
                b[j] += (t[j][i % 10] - saida)
                
                print(f"Os novos pesos para este neuronio sao demonstrados abaixo:")
                print("Pesos:", w[j])
                print("Bias:", b[j])
                print(f"As contas a partir da primeira entrada possivel serao refeitas para nova verificacao de sucesso ou nao deste treinamento.")

                i = 0
            else:
                i+=1
        print(f"Nao constando mais nenhum erro com a lista de pesos atual deste neuronio, o treinamento deste e dado como finalizado.")
        print("Pesos:", w[j])
        print("Bias:", b[j])
    
    print("\nOs pesos apos o fim do treinamento da rede neural como um todo foram setados como:")
    print(w)

    print("\nTem-se a lista de bias abaixo:")
    print(b)

    sys.stdout = sys.__stdout__

print("Arquivo relatorio.txt criado com sucesso.")