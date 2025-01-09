import numpy as np
import pandas as pd

# Parâmetros do ACO
num_ants = 853
alpha = 1.0
beta = 2.0
rho = 0.1
Q = 1000
num_iterations = 2000

# Função para completar a matriz inferior a partir da matriz superior
def complete_symmetric_matrix(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[j][i] = matrix[i][j]
    return matrix

# Leitura da matriz de distâncias do arquivo CSV
def read_distance_matrix_from_csv(file_path):
    # Ignora a primeira linha e as duas primeiras colunas
    matrix = pd.read_csv(file_path, header=0).iloc[0:, 2:].to_numpy()
    return complete_symmetric_matrix(matrix)

# Leitura da matriz e definição do número de locais
file_path = r'C:\Users\ceoba\Downloads\dados_1_853.csv' # Insira o caminho do seu arquivo CSV
distance_matrix = read_distance_matrix_from_csv(file_path)
num_locations = distance_matrix.shape[0]

# Inicialização do feromônio
pheromone_matrix = np.ones((num_locations, num_locations))

# Função para calcular as probabilidades de transição
def calculate_probabilities(current_location, unvisited, pheromone, distance):
    probs = []
    for loc in unvisited:
        pheromone_level = pheromone[current_location][loc] ** alpha
        heuristic_value = (1 / distance[current_location][loc]) ** beta
        probs.append(pheromone_level * heuristic_value)
    probs = np.array(probs)
    return probs / probs.sum()

# Algoritmo de Otimização por Colônia de Formigas (ACO)
def ant_colony_optimization():
    global pheromone_matrix
    best_path = None
    best_path_length = float('inf')

    with open("aco_results.txt", "w") as file:
        for iteration in range(num_iterations):
            paths = []
            path_lengths = []

            for ant in range(num_ants):
                unvisited = list(range(num_locations))
                path = []
                current_location = np.random.choice(unvisited)
                path.append(current_location)
                unvisited.remove(current_location)

                while unvisited:
                    probs = calculate_probabilities(current_location, unvisited, pheromone_matrix, distance_matrix)
                    next_location = np.random.choice(unvisited, p=probs)
                    path.append(next_location)
                    unvisited.remove(next_location)
                    current_location = next_location

                path.append(path[0])
                paths.append(path)
                path_length = sum(distance_matrix[path[i]][path[i+1]] for i in range(len(path) - 1))
                path_lengths.append(path_length)

                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path

            pheromone_matrix *= (1 - rho)
            for path, length in zip(paths, path_lengths):
                for i in range(len(path) - 1):
                    pheromone_matrix[path[i]][path[i+1]] += Q / length
                    pheromone_matrix[path[i+1]][path[i]] += Q / length

            # Salvar detalhes a cada 10 iterações
            if (iteration + 1) % 10 == 0:
                file.write(f"Iteração {iteration + 1}:\n")
                file.write("Matriz de Feromônios:\n")
                np.savetxt(file, pheromone_matrix, fmt="%.6f")
                file.write(f"Melhor Caminho: {best_path}\n")
                file.write(f"Custo do Melhor Caminho: {best_path_length}\n")
                file.write("-" * 50 + "\n")

    return best_path, best_path_length

# Executar o ACO
best_path, best_path_length = ant_colony_optimization()
best_path_converted = [node + 1 for node in best_path]

print("Melhor caminho encontrado:", best_path_converted)
print("Comprimento do melhor caminho:", best_path_length)
