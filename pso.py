import numpy as np
import random
from sympy import symbols, simplify, Symbol, Function, lambdify
from sympy import *


def reconhece_funcao(expressao):
    
    x = symbols('x')
    y = symbols('y')
    
    try:
        # Tenta simplificar a expressão fornecida
        simplificada_expressao = simplify(expressao)
        dfdx = simplificada_expressao.diff(x)
        dfdy = simplificada_expressao.diff(y)
        
        print("f = ",simplificada_expressao,"\n", 
              "dfdx = ", dfdx,"\n",
              "dfdy = ", dfdy)
        
    except:
        print('Erro de formatação')
    
    return lambdify([x, y], simplificada_expressao, 'numpy'), lambdify([x, y], dfdx, 'numpy'), lambdify([x, y], dfdy, 'numpy')
 
 
def reconhece_restricao(expressao):
    
    x = symbols('x')
    y = symbols('y')
    
    try:
        # Tenta simplificar a expressão fornecida
        simplificada_expressao = simplify(expressao)
        print("sujeito a: ",simplificada_expressao,"\n")
       
    except:
        print('Erro de formatação')
    
    return lambdify([x, y], simplificada_expressao, 'numpy')
 
 
def PSO(funcaoObjetivo, restricoes):
    
    def define_gbest(populacao):
        
        gbest_custo = 100000 #numero deve ser alto para minimização
        for particula in populacao:
            if particula.pbest_custo < gbest_custo:
                gbest_custo = particula.pbest_custo
                gbest_posicao = particula.pbest_posicao
                        
        return gbest_custo, gbest_posicao
        
    
    def valida_posicoes(posicao):

        if not all([rest(posicao[0], posicao[1]) for rest in restricoes]):
            return True
        else:
            return False


    def valida_posicoes_iniciais(posicao_x, posicao_y):

        #print("x: ",posicao_x,"y: ", posicao_y)
        #print(not all([rest(posicao_x, posicao_y) for rest in restricoes]))
        while not all([rest(posicao_x, posicao_y) for rest in restricoes]):
            posicao_x = np.random.uniform(xyMin, xyMax)
            posicao_y = np.random.uniform(xyMin, xyMax)

        print("x: ",posicao_x,"y: ", posicao_y)
        
        return [posicao_x,  posicao_y]

    def atualiza_V(populacao, w, c1, c2):
        for particula in populacao.matrix:
            particula.velocidade = w*np.array(particula.velocidade) + c1*random.random()*(np.array(particula.pbest_posicao) - np.array(particula.posicao)) + c2*random.random()*(np.array(populacao.gbest_posicao) - np.array(particula.posicao))
        return populacao
    
    
    def atualiza_X(populacao):
        
        for particula in populacao.matrix:
            particula.posicao = np.array(particula.posicao) + np.array(particula.velocidade)
            particula.posicao_x = particula.posicao[0]
            particula.posicao_y = particula.posicao[1]
            
        return populacao
    
    
    def avaliar(populacao):
        
        #avaliação de pariculas (atualização de pbest e pbest_custo)
        for particula in populacao.matrix:
            particula.custo = funcaoObjetivo(particula.posicao_x, particula.posicao_y)
            if particula.pbest_custo > particula.custo and valida_posicoes([particula.posicao_x, particula.posicao_y]):
                particula.pbest_custo = particula.custo
                particula.pbest_posicao = particula.posicao
                
        
        #avaliação de enxame (atualização de gbest e gbest_custo)
        for particula in populacao.matrix:
            if populacao.gbest_custo > particula.pbest_custo:
                populacao.gbest_custo = particula.pbest_custo
                populacao.gbest_posicao = particula.pbest_posicao
        
        return populacao
    
    
    #Definição de variaveis 
    MaxIt = 2000 #numero de maximo iterações
    xyMin, xyMax = -100, 100 #limites numéricos do espaço xy
    vMin, vMax = -0.2*(xyMax - xyMin), 0.2*(xyMax - xyMin) #limites de velocidade
    ps = 10 #tamanho da população
    w = 0.9 - ((0.9 - 0.4)/MaxIt) #* np.linspace(0,MaxIt, MaxIt) #inercia
    c1, c2 = 1, 2 #constantes
    
    class Particula():
        
        def __init__(self, id):
            
            self.id = id
            self.posicao_x = np.random.uniform(xyMin, xyMax)
            self.posicao_y = np.random.uniform(xyMin, xyMax)
            self.posicao = valida_posicoes_iniciais(self.posicao_x, self.posicao_y)
            self.velocidade = [np.random.uniform(vMin, vMax), np.random.uniform(vMin, vMax)]
            self.custo = funcaoObjetivo(self.posicao_x, self.posicao_y) #fitness
            self.pbest_posicao = [self.posicao_x, self.posicao_y]
            self.pbest_custo = self.custo
            
    class Enxame():
        
        def __init__(self, populacao):
            
            self.matrix = populacao
            self.gbest_custo, self.gbest_posicao = define_gbest(self.matrix)
        
    


    #Inicialização
    populacao = []
    
    for i in range(ps):
        individuo = Particula(id = i)
        populacao.append(individuo)
    
    #print(populacao)  
    populacao = Enxame(populacao = populacao)
    
    #print(populacao)
    #print(particula.pbest for particula in populacao.matrix)
    #print(populacao.gbest_posicao, populacao.gbest_custo)
    
    #Iterações com atualização de posições e velocidades
    gbest = 100000
    for i in range(MaxIt):
        populacao = atualiza_V(populacao, w, c1, c2)
        populacao = atualiza_X(populacao)
        populacao = avaliar(populacao)
        if gbest > populacao.gbest_custo:
            gbest = populacao.gbest_custo
            print("iteração", i, "melhor resultado:", gbest, "posição:", populacao.gbest_posicao)
        
    #Print gbest
    print('melhor resultado:', populacao.gbest_custo)
    print('posição:', populacao.gbest_posicao)
    return       
    
#Input de função a ser otimizada
x,y,z = symbols('x y z')
funcaoObjetivo_input = input("Digite a função a ser otimizada: ")
qtde_restricoes = int(input("Digite quantidade de restrições: "))
restricoes = []
for i in range(qtde_restricoes):
    aux = "Restrição "+str(i+1)+":"
    restricoes.append(input(aux))
#funcaoObjetivo_input  = "x^2 + y^2"
funcaoObjetivo, dfdx, dfdy = reconhece_funcao(funcaoObjetivo_input)
restricoes_tratadas = [reconhece_restricao(rest) for rest in restricoes]



PSO(funcaoObjetivo=dfdx, restricoes = restricoes_tratadas)
