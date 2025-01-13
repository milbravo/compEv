import numpy as np
import pandas as pd
import time
import pickle
import os

# Leitura da matriz de distâncias do arquivo CSV
def ler_matriz_do_arquivo(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    matriz = df.values
    return matriz

# Função de avaliação (distância total do caminho)
def funcao_objetivo(caminho, matriz_distancias):
    distancia_total = 0
    for i in range(len(caminho) - 1):
        distancia_total += matriz_distancias[caminho[i], caminho[i + 1]]
    distancia_total += matriz_distancias[caminho[-1], caminho[0]]  # Retorno à cidade inicial
    return distancia_total

# Função para gerar uma solução inicial usando o Algoritmo do Vizinho Mais Próximo
def generate_initial_solution(distance_matrix):
    n = len(distance_matrix)
    start = np.random.randint(0, n)  # Escolhe uma cidade inicial aleatória
    unvisited = set(range(n))
    unvisited.remove(start)
    solution = [start]

    while unvisited:
        current = solution[-1]
        next_city = min(unvisited, key=lambda city: distance_matrix[current][city])
        solution.append(next_city)
        unvisited.remove(next_city)

    return solution

# Função para gerar vizinhança 2-opt
def generate_neighborhood_2opt(solution):
    neighbors = []
    n = len(solution)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i != 0 or j != n - 1:  # Evita trocar a cidade inicial
                new_solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]
                neighbors.append(new_solution)
    return neighbors

# Função para salvar a lista tabu
def save_tabu_list(tabu_list, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(tabu_list, file)

# Função para carregar a lista tabu
def load_tabu_list(filepath, distance_matrix):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as file:
                tabu_list = pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            best_solution = generate_initial_solution(distance_matrix)
            tabu_list = [best_solution]
    else:
        best_solution = generate_initial_solution(distance_matrix)
        tabu_list = [best_solution]
    return tabu_list

# Implementação da Busca Tabu com Critério de Aspiração
def tabu_search(distance_matrix, iterations, initial_tabu_size, max_tabu_size, local_optimum_period, iteration, best_solution, best_cost, start_time, tabu_filepath):
    n = len(distance_matrix)

    # Verifica se uma solução inicial foi fornecida, caso contrário gera uma nova solução inicial
    if best_solution is None or best_solution == []:
        best_solution = generate_initial_solution(distance_matrix)
        best_cost = funcao_objetivo(best_solution, distance_matrix)

    # Inicializa a solução atual com a melhor solução encontrada até agora
    current_solution = best_solution
    current_cost = best_cost

    # Carrega a lista tabu
    tabu_list = load_tabu_list(tabu_filepath, distance_matrix)
    tabu_size = initial_tabu_size
    local_optimum_counter = 0

    # Abre o arquivo para escrita
    with open('resultados.txt', 'w') as f:
        for iteration in range(iteration, iterations):
            # Gera vizinhanças da solução atual
            neighborhood = generate_neighborhood_2opt(current_solution)

            # Avalia as soluções vizinhas
            neighborhood_costs = [(funcao_objetivo(neighbor, distance_matrix), neighbor) for neighbor in neighborhood]
            # Ordena as soluções pelo custo
            neighborhood_costs.sort(key=lambda x: x[0])

            # Encontra a melhor solução que não está na lista tabu ou que atende ao critério de aspiração
            for cost, solution in neighborhood_costs:
                if solution not in tabu_list or cost < best_cost:  # Critério de aspiração
                    current_solution = solution
                    current_cost = cost
                    break

            # Atualiza a melhor solução encontrada
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                local_optimum_counter = 0  # Reseta o contador de ótimo local
                tabu_size = initial_tabu_size  # Volta ao tamanho inicial da lista tabu
            else:
                local_optimum_counter += 1
                if local_optimum_counter > local_optimum_period:
                    tabu_size = min(tabu_size + 1, max_tabu_size)  # Aumenta o tamanho da lista tabu até o máximo permitido

            # Atualiza a lista tabu
            tabu_list.append(current_solution)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            # Escreve os resultados e a lista tabu no arquivo a cada 10 iterações
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time
                f.write(f"Iteração {iteration + 1}:\n")
                f.write("Melhor solução até agora: " + str(best_solution) + "\n")
                f.write("Custo da melhor solução até agora: " + str(best_cost) + "\n")
                f.write(f"Tempo decorrido: {elapsed_time:.2f} segundos\n")
                f.flush()  # Garante que os dados sejam escritos no arquivo imediatamente

                # Salva a lista tabu a cada 10 iterações
                save_tabu_list(tabu_list, tabu_filepath)

    return best_solution, best_cost

# Leitura da matriz de distâncias
def complete_symmetric_matrix(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[j][i] = matrix[i][j]
    return matrix

def read_distance_matrix_from_csv(file_path):
    matrix = pd.read_csv(file_path, header=0).iloc[0:, 2:].to_numpy()
    return complete_symmetric_matrix(matrix)

if __name__ == '__main__':
    # Caminho para o arquivo CSV
    file_path = r'c:/Users/marce/OneDrive/Desktop/busca tabu/dados_1_853.csv'
    distance_matrix = read_distance_matrix_from_csv(file_path)

    # Definição dos parâmetros 
    num_iterations = 2000  # Número de iterações
    initial_tabu_size = 50  # Tamanho inicial da lista tabu
    max_tabu_size = 100  # Tamanho máximo da lista tabu
    local_optimum_period = 200  # Período de ótimos locais

    iteration = 0  # última iteração
    best_cost = 0  # melhor custo encontrado atual
    start_time = time.time()  # Defina o tempo de início atual
    best_solution = None  # melhor solução atual
    tabu_filepath = 'tabu_list.pkl'
    
    # Execução da Busca Tabu
    best_solution, best_cost = tabu_search(distance_matrix, num_iterations, initial_tabu_size, max_tabu_size, local_optimum_period, iteration, best_solution, best_cost, start_time, tabu_filepath)
    
    print("Melhor solução encontrada:", best_solution)
    print("Custo da melhor solução encontrada:", best_cost)
