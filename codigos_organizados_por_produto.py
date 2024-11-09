
# Produto: Modelo 1 (Mod1)
# Código para gerar o gráfico Mod1 com a configuração específica
def gerar_grafico_mod1(coluna_dados, labels):
    import matplotlib.pyplot as plt
    
    # Configurações gerais do gráfico Mod1
    plt.figure(figsize=(5, 7))
    plt.plot(coluna_dados, labels, marker='o', color='purple', linestyle='-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Valores")
    plt.ylabel("Subníveis")
    plt.grid(visible=True, linestyle="--", linewidth=0.5)
    plt.title("Gráfico Mod1")
    plt.show()

# Exemplo de uso
# gerar_grafico_mod1(dados_coluna, niveis)

# Produto: Modelo 2 (Mod2)
# Código para gerar o gráfico Mod2 comparando duas colunas
def gerar_grafico_mod2(coluna_dados1, coluna_dados2, labels, legenda1="Inteiros", legenda2="Fragmentados"):
    import matplotlib.pyplot as plt
    
    # Configurações gerais do gráfico Mod2
    plt.figure(figsize=(5, 7))
    plt.plot(coluna_dados1, labels, marker='o', color='blue', linestyle='-', label=legenda1)
    plt.plot(coluna_dados2, labels, marker='o', color='red', linestyle='-', label=legenda2)
    plt.gca().invert_yaxis()
    plt.xlabel("Valores")
    plt.ylabel("Subníveis")
    plt.legend()
    plt.grid(visible=True, linestyle="--", linewidth=0.5)
    plt.title("Gráfico Mod2")
    plt.show()

# Exemplo de uso
# gerar_grafico_mod2(dados_coluna1, dados_coluna2, niveis)

# Produto: Modelo 3 (Mod3)
# Código para gerar valores de IDW baseados na profundidade e organizar os dados por profundidade
def calcular_idw(profundidades, valores, profundidade_interpolada):
    import numpy as np
    
    # Inverse Distance Weighting (IDW)
    pesos = 1 / np.abs(np.array(profundidades) - profundidade_interpolada)
    valores_interpolados = np.sum(valores * pesos) / np.sum(pesos)
    return valores_interpolados

# Exemplo de uso
# profundidades_existentes = [10, 20, 30, ...]
# valores_existentes = [15.0, 18.2, 19.6, ...]
# profundidade_desejada = 35
# valor_idw = calcular_idw(profundidades_existentes, valores_existentes, profundidade_desejada)

# Produto: Modelo 4 (Mod4)
# Código para gerar gráficos com profundidade no eixo Y, aplicando configurações do Mod1
def gerar_grafico_mod4(coluna_dados, profundidades):
    import matplotlib.pyplot as plt
    
    # Configurações gerais do gráfico Mod4
    plt.figure(figsize=(5, 7))
    plt.plot(coluna_dados, profundidades, marker='o', color='purple', linestyle='-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Valores")
    plt.ylabel("Profundidade (cm)")
    plt.grid(visible=True, linestyle="--", linewidth=0.5)
    plt.title("Gráfico Mod4")
    plt.show()

# Exemplo de uso
# gerar_grafico_mod4(dados_coluna, profundidades)
