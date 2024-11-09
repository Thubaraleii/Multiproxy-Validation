
import numpy as np
from scipy.interpolate import griddata

# Dados fornecidos na imagem para interpolação
data_elements = {
    "Profundidade (cm)": [10, 35, 45, 60, 68, 72, 76, 84, 89, 95, 99, 101, 102],
    "Al2O3": [None, None, 14.13, 14.31, 13.69, 14.1, 13.62, 15.01, 14.63, 13.95, 14.89, 13.02, 13.12],
    "Fe2O3": [None, None, 7.06, 6.46, 6.41, 5.23, 7.41, 5.99, 6.26, 5.76, 5.25, 4.94, 4.92],
    "Cr2O3": [None, None, 0.039, 0.064, 0.037, 0.038, 0.037, 0.034, 0.032, 0.016, 0.019, 0.028, 0.028],
    "Cr": [None, None, 266.84, 369.47, 253.16, 260, 253.16, 164.21, 164.21, 109.47, 91.58, 191.58, 191.58]
}

# Convertendo para DataFrame
df_elements = pd.DataFrame(data_elements)

# Profundidades conhecidas e colunas com dados conhecidos
known_depths = df_elements["Profundidade (cm)"].dropna()
columns_to_interpolate = ["Al2O3", "Fe2O3", "Cr2O3", "Cr"]

# Ponto de profundidade onde queremos interpolar (35 cm e 10 cm)
target_depths = [35, 10]

# Função para interpolação IDW
def interpolate_idw(column, known_depths, target_depths):
    known_values = df_elements[column].dropna()
    known_positions = known_depths[~df_elements[column].isna()]
    return griddata(known_positions.to_numpy(), known_values.to_numpy(), target_depths, method='nearest')

# Aplicando interpolação para cada coluna
interpolated_values = {depth: {} for depth in target_depths}
for column in columns_to_interpolate:
    interpolated_column_values = interpolate_idw(column, known_depths, target_depths)
    for i, depth in enumerate(target_depths):
        interpolated_values[depth][column] = interpolated_column_values[i]

# Apresentando os valores interpolados
interpolated_df = pd.DataFrame(interpolated_values).T
import ace_tools as tools; tools.display_dataframe_to_user(name="Interpolated Values for 4B and 4C", dataframe=interpolated_df)

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
