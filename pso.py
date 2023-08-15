import numpy as np


def funcaoObjetivo(vars):
    N = len(vars) - 1
    soma_x = 0
    soma_cos = 0
    
    for i in range(N):
        soma_x += vars[i] ** 2
        soma_cos += np.cos(2 * np.pi * vars[i])
    
    fit = -20 * np.exp(-0.2 * np.sqrt((1/N) * soma_x)) - np.exp((1/N) * soma_cos) + 20 + np.exp(1)
    return fit

# Valores de exemplo para teste
vars_example = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Substitua pelos valores desejados
resultado = funcaoObjetivo(vars_example)
print("Resultado da função objetivo:", resultado)


def PSO(dim, nPop, vInit, nIter):
    w = 0.1
    c1 = 1
    c2 = 1

    swarm = np.random.randint(vInit[0], vInit[1] + 1, size=(nPop, dim + 1))
    # A última coluna contém o valor de fitness
    swarm_melhores = np.zeros((nPop, dim + 1))
    velocidade = np.zeros((nPop, dim))
    todos_melhores = np.zeros(nIter)

    # Inicializando fitness com zero
    swarm[:, dim] = 0

    melhor, vPos, swarm = fitness_init(swarm, nPop, dim)
    swarm_melhores = swarm.copy()

    for i in range(nIter):
        melhor, vPos, swarm, swarm_melhores = fitness(swarm, swarm_melhores, melhor, vPos, nPop, dim)
        velocidade = calc_velocidade(swarm, swarm_melhores, velocidade, melhor, vPos, nPop, dim, w, c1, c2)
        swarm = posicao(swarm, velocidade, nPop, dim)
        todos_melhores[i] = melhor

    pos = swarm_melhores[vPos, :dim]
    return melhor, pos, todos_melhores

def fitness_init(swarm, nPop, dim):
    melhor = funcaoObjetivo(swarm[0, :])  # apenas para fins de inicialização
    vPos = 0
    for i in range(nPop):
        fit = funcaoObjetivo(swarm[i, :])
        if fit < melhor:
            melhor = fit
            vPos = i
        swarm[i, dim] = fit
    return melhor, vPos, swarm

def fitness(swarm, swarm_melhores, melhor, vPos, nPop, dim):
    for i in range(nPop):
        swarm[i, dim] = funcaoObjetivo(swarm[i, :])
        if swarm_melhores[i, dim] > swarm[i, dim]:
            swarm_melhores[i, :] = swarm[i, :]
            swarm_melhores[i, dim] = swarm[i, dim]
            if swarm_melhores[i, dim] < melhor:
                melhor = swarm_melhores[i, dim]
                vPos = i
    return melhor, vPos, swarm, swarm_melhores

def calc_velocidade(swarm, swarm_melhores, velocidade, melhor, vPos, nPop, dim, w, c1, c2):
    for i in range(nPop):
        velocidade[i, :] = w * velocidade[i, :] + c1 * np.random.rand() * (swarm_melhores[i, :dim] - swarm[i, :dim]) + c2 * np.random.rand() * (swarm_melhores[vPos, :dim] - swarm[i, :dim])
    return velocidade

def posicao(swarm, velocidade, nPop, dim):
    for i in range(nPop):
        swarm[i, :dim] = swarm[i, :dim] + velocidade[i, :]
    return swarm

# Sua função objetivo deve ser definida em um arquivo separado com o nome 'funcaoObjetivo.py'
# Implemente a função funcaoObjetivo nas linhas abaixo

# from funcaoObjetivo import funcaoObjetivo

# Valores de exemplo para teste
dim = 10
nPop = 20
vInit = [-32, 32]
nIter = 100

melhor, pos, todos_melhores = PSO(dim, nPop, vInit, nIter)
print("Melhor resultado:", melhor)
print("Posição do melhor resultado:", pos)
print("Histórico dos melhores resultados:", todos_melhores)
