import numpy as np
import pandas as pd
import random
import json


# Função para completar a matriz inferior a partir da matriz superior
def complete_symmetric_matrix(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i==j:
              matrix[i][j]=100000000
            else:
              matrix[j][i] = matrix[i][j]
    return matrix

# Leitura da matriz de distâncias do arquivo CSV
def read_distance_matrix_from_csv(file_path):
    # Ignora a primeira linha e as duas primeiras colunas
    matrix = pd.read_csv(file_path, header=0).iloc[0:, 2:].to_numpy()
    return complete_symmetric_matrix(matrix)

# Função para carregar a matriz pbest
def load_pbest_matrix(file_path):
    with open(file_path, "r") as file:
        matrix = []
        for line in file:
            matrix.append(list(map(float, line.strip().split())))
    return np.array(matrix)

# Função para calcular a distância total de uma permutação
def calculate_distance(permutation, distance_matrix):
    return (sum(distance_matrix[permutation[i], permutation[i + 1]] for i in range(len(permutation) - 1))
                + distance_matrix[permutation[len(permutation)-1], permutation[0]])  # Retorno à cidade inicial

# Operação para gerar uma nova permutação (troca de posições)
def swap_positions(permutation, i,j):
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation

# Função para salvar os dados em um arquivo JSON, acrescentando ao conteúdo existente
def append_to_json(filename, global_best, global_best_cost, iteration):
    try:
        # Tenta carregar os dados existentes no arquivo
        with open(filename, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []  # Garante que os dados sejam uma lista
    except (FileNotFoundError, json.JSONDecodeError):
        data = []  # Se o arquivo não existe ou está vazio, inicia uma lista vazia

    # Novo registro de dados
    new_entry = {
        "iteration": int(iteration),
        "global_best": list(global_best),  # Converte cada permutação para lista de ints
        "global_best_cost": int(global_best_cost),      # Converte custos globais para floats
    }

    # Acrescenta o novo registro aos dados existentes
    data.append(new_entry)

    # Salva os dados atualizados de volta no arquivo
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# Algoritmo PSO para o TSP
def pso_tsp(distance_matrix, num_particles,W, C1, C2, max_iterations, max_trocas):
    num_cities = len(distance_matrix)
    save_every=10000
    save_file="PSO_results.json"
    # Inicializar partículas (permutação aleatória das cidades)
    particles = [np.random.permutation(num_cities).tolist() for _ in range(num_particles)]
    velocities = np.ones(num_particles)  # Velocidades não são diretamente usadas, mas simuladas com trocas

    # Melhores locais e global
    personal_best = particles[:]
    personal_best_cost = [calculate_distance(p, distance_matrix) for p in particles]
    global_best = list()
    global_best.append(personal_best[np.argmin(personal_best_cost)])
    global_best_cost = list()
    global_best_cost.append(min(personal_best_cost))

    for iteration in range(max_iterations):
      for ip in range(num_particles):
        # Atualizar a partícula usando operadores combinatórios

        current_particle_cost=calculate_distance(particles[ip], distance_matrix)
        new_particle = particles[ip][:]  # Copiar partícula atual
        #Atualiza a velocidade
        Vpbest=C1*(np.random.rand())*((personal_best_cost[ip]-current_particle_cost))
        Vgbest=C2*(np.random.rand())*((global_best_cost[-1]-current_particle_cost))
        velocities[ip]=(W*velocities[ip]+Vpbest+Vgbest)
        
        ipbest=abs(int(100*(1-(velocities[ip]-Vpbest)/(velocities[ip]))))
        igbest=abs(int(100*(1-(velocities[ip]-Vgbest)/(velocities[ip]))))

        #Trava o númeor maximo de trocas
        if(igbest>=max_trocas):
          igbest=max_trocas
        if(ipbest>=max_trocas):
          ipbest=max_trocas
        
        '''if(ip==1):
          print("Velocity: ",velocities[ip])
          print("Vpbest: ",Vpbest)
          print("Vgbest: ",Vgbest)
          print("ipbest: ",ipbest)
          print("igbest: ",igbest)
          print("Gbest: ", global_best_cost[-1])

        for i in range(3):
            m, k = random.sample(range(num_cities), 2)
            new_particle=swap_positions(new_particle,m,k)  ''' # Aplicar uma troca em si mesma
        for i in range(ipbest):
          city=np.random.randint(0,853)
          m = (personal_best[ip]).index(city)
          k = new_particle.index(city)
          new_particle=swap_positions(new_particle,k,m)   # Aplicar uma troca em relaçao ao personal best
        for i in range(igbest):
          city=np.random.randint(0,853)
          m = (global_best[-1]).index(city)
          k = new_particle.index(city)
          new_particle=swap_positions(new_particle,k,m)   # Aplicar uma troca em relaçao ao global best

        if(abs(velocities[ip])<0.0001): #Caso todas as particulas fiquem iguais
          velocities[ip]=1
          for i in range(2):
            m, k = random.sample(range(num_cities), 2)
            new_particle=swap_positions(new_particle,m,k)   # Aplicar uma troca em si mesma

        particles[ip] = new_particle
        new_particle_cost=calculate_distance(new_particle, distance_matrix)
        if new_particle_cost < personal_best_cost[ip]:
            personal_best[ip] = new_particle # Atualizar o melhor local
            personal_best_cost[ip] = new_particle_cost # Atualizar custo do melhor local

      # Acrescenta no melhor global

      min_cost_index = np.argmin(personal_best_cost)
      if(min_cost_index<global_best_cost[-1]):
        global_best.append(personal_best[min_cost_index][:])
        global_best_cost.append(personal_best_cost[min_cost_index])
      # Salvar a cada 'save_every' iterações

      if (iteration + 1) % save_every == 0:
          append_to_json(save_file, global_best[-1], global_best_cost[-1], iteration + 1)
          print("Gbest: ", global_best_cost[-1])

    return global_best[-1], global_best_cost[-1]


# Leitura da matriz e definição do número de locais
file_path_data = 'https://raw.githubusercontent.com/milbravo/compEv/refs/heads/main/dados_1_853.csv'
distance_matrix = read_distance_matrix_from_csv(file_path_data)

w=0.6
c1=2
c2=2

# Executar PSO
best_path, best_cost= pso_tsp(distance_matrix,50,w,c1,c2,1000000,20)
print("Melhor caminho:", best_path)
print("Custo do melhor caminho:", best_cost)