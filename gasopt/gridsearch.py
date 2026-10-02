import itertools
import numpy as np
import multiprocessing
import warnings
import pandas as pd
from gasopt import GasMix
from math import comb

from heapq import heapify
from heapq import heappush as push
from heapq import heappop as hpop
from heapq import heapreplace as replace

# This class calculates the the kappa for every possible combination
# of elements of a sample. Then find a N size smallest kappas set   

# Para não estorar o limite de memória, ou utilizar memória 
# demais (deixando o pc lerdo), 

# Para manter o consumo constante de memória vou salvar apenas as
# N melhores combinações em um max heap. Tambem vou dividir as 
# combinações em pedaços (chunks).
# Então vou calcular os kappas em um chunk, 


import json
from pathlib import Path
from dispersion import Gas
# espera essa estruta de arquivo
#
# gas_refractometry_optimize
# ├── gasopt
# │   ├── __init__.py
# │   ├── gridsearch.py
# ├── data
# │   └── gas_database.json

# Caminho do arquivo atual
current_file = Path(__file__).resolve()

# .parent é 'scripts/', .parent.parent é a raiz do projeto 'gas_refractometry_optimize/'
BASE_DIR = current_file.parent.parent 

# caminho do database
DATAPATH = BASE_DIR / 'data' / 'gas_database.json'
with open(DATAPATH, 'r') as f:
    DATABASE = json.load(f)

def ids_hash(gasesid):
    gashash = dict()
    # encontrando os ids validos para cada gas
    for gas in gasesid:
        valid_id = True 
        gashash[gas] = []
        i = 1
        while valid_id: 
            try:
                curr_id = f'{gas}_{i}'
                DATABASE[curr_id]
                gashash[gas] += [Gas(id=curr_id)]
            except:
                valid_id = False

            i += 1

    return gashash

class GridSearch():
    def __init__(self, sample: np.ndarray = np.array([]), k: int = 4):
        '''
        Params:
        sample: np.array, sample space for the combinations, ie
                set of wavelenghts in micrometers 
        k: int, size of each combination 
        '''
        
        #
        self.sample = sample
        self.k = k

        # estou contando com o fato que o na minha bases os 
        # gases são salvos com um id que é o nome + indice
        # ex: co2_1, co2_2
        # entao vou salvar o gases sem o indice, e escolher o 
        # indice depois, baseado na validade de cada equação
        #
        # depois pensar em como o usuario escolhe isso
        self.gasesid = ['co2', 'o2', 'ar', 'n2']
        
        self.gasesid_hash = ids_hash(self.gasesid)

        # temperatura da mistura
        self.temp = 273.15

        # tamanho do grid_search
        # n escolhe k, (n k)
        self.size = comb(len(self.sample), self.k)

        # inicializa um iteravel com as combinações
        # as combinações serão feitas com os índeices 
        # de sample
        idx = range(len(self.sample))
        self.combs = itertools.combinations(idx, self.k)

        # max heap com as melhores 
        self.heap = []
        heapify(self.heap)

    def __len__(self):
        return self.size

    def _get_chunck(self, size: int = 0):
        '''
        Escolhe um conjunto de tamanho size de combinações ainda não testadas

        Se restarem menos combinações que o tamanho size, retorn apenas as 
        as combinações que faltam  
        '''
        return list(itertools.islice(self.combs, size))

    def _gas_valid_interval(self, gasid, wvlen):
        '''
        Encontra uma equação valida para o gas de id = gasid
        para o comprimento de onda wvlen (em um).
        Se nao encontrar, avisa com warning e usa o primeiro candidato.
        '''
        candidatos = self.gasesid_hash[gasid]
        for g in candidatos:
            rmin, rmax = g.range
            if rmin <= wvlen <= rmax:
                return g
        
        # se nenhuma equacao cobrir esse comprimento de onda, avisa e usa o primeiro
        warnings.warn(f"Comprimento de onda {wvlen:.4f} um fora da faixa para '{gasid}'. Usando extrapolacao de {candidatos[0].name}.")
        return candidatos[0]

    def _get_bigmatrix(self):
        '''
        Calcula de antemão todas as refratividades. Isso é, refratividade
        para todos os comprimentos de onda e todos os gases.
        
        Para cada lambda e para cada gas, escolhe a equacao mais adequada.
        '''
        self.bigmatrix = np.zeros((len(self.sample), len(self.gasesid)))

        for i, wl in enumerate(self.sample):
            for j, gas in enumerate(self.gasesid):
                g = self._gas_valid_interval(gas, wl)
                # fator de temperatura
                t_factor = g.t_ref / self.temp
                refrac = g.calculate_sellmeier(float(wl))[0]
                self.bigmatrix[i, j] = refrac * t_factor

    def _calculate_kappa(self, chunk):
        '''
        chunk é a matriz de índices, para um combinção específica
        índices relativos a bigmatrix  
        '''


        # tensor tamanho: M x N x chunksize
        # recorta a matriz da refratividades para aquelas 
        # que estão nesse chunk
        tensor = self.bigmatrix[chunk]

        # tensor dos valores singulares
        s = np.linalg.svd(tensor, compute_uv=False)
        
        # o numero de condição pode ser definido como a razao
        # k(A) = M/m
        # M: max que A estica um vetor x
        # m: min que A comprime um vetor x
        # Ao mesmo tempo, isso também é exatamente a razão entre
        # o menor valor singular de A com o maior
        kappas = s[:, 0] / s[:, -1]
        return kappas

    def _update_heap(self, kappas, chunk):
        '''
        Recebe uma lista de candidatos kappas e verifica se eles são
        bons ou ruins.

        Kappas menores que o maior elemento do heap são salvos, e 
        kappas menores esquecidos.

        O heap é mantido com tamanho fixo
        '''

        for k, idx in zip(kappas, chunk):
            lambdas = self.sample[list(idx)]
            # enche o heap até o tamanho 
            if len(self.heap) < self.nbest:
                push(self.heap, (-k, lambdas))
            else:
                # O heapq é implementado como um min heap
                # então estou usando os opostos (*-1)
                if -k > self.heap[0][0]:
                    replace(self.heap, (-k, lambdas))

    def _heap2array(self):
        '''
        Transforma o heap em uma array
        '''
        n_items = len(self.heap)
        arr = np.zeros((n_items, self.k + 1))
        
        # percorre a lista de tras para frente
        # para ter um ordenação crescente
        for i in range(n_items-1, -1, -1):
            k, lambdas = hpop(self.heap)
            
            arr[i, 0] = -k # k esta invertido no heap
            arr[i, 1:] = lambdas

        return arr

    def run_seach(self, n_best: int = 100, chunk_size: int = 1000):
        '''
        n_best: int, numero do conjunto das melhores combinações
        '''
        
        self.nbest = n_best

        # cria uma matrix M x N
        # M: Numero de gases 
        # N: Numero de lasers
        # aqui estão todas as refratividades possíveis já
        self._get_bigmatrix()

        for i in range(self.size // chunk_size + 1):
            curr_chunk = self._get_chunck(chunk_size)
            curr_kappas = self._calculate_kappa(curr_chunk)
            self._update_heap(curr_kappas, curr_chunk)            

        
        return GridResult(
            arr = self._heap2array(),
            k = self.k,
            sample = self.sample,
            grid=self
        )

class GridResult:
    def __init__(self, arr, k, sample, grid = None):
        self.raw = arr
        self.k = k
        self.grid = grid
        # O melhor é a linha 0
        self.best_kappa = arr[0, 0]
        self.best_lambdas = arr[0, 1:]
        
        
    def to_dataframe(self):
        cols = ['kappa'] + [f'l_{i+1}' for i in range(self.k)]
        df = pd.DataFrame(self.raw, columns=cols)
        
        # se tiver o buscador, descobre sob demanda os ids usados
        if self.grid is not None:
            for i in range(self.k):
                col_laser = f'l_{i+1}'
                col_ids = f'l_{i+1}_ids'
                
                # para cada linha do dataframe, busca os ids dos gases nesse lambda
                ids_coluna = []
                for wl in df[col_laser]:
                    ids_laser = {
                        str(self.grid._gas_valid_interval(gas, wl))
                        for gas in self.grid.gasesid
                    }
                    ids_coluna.append(ids_laser)
                
                df[col_ids] = ids_coluna
                
        return df

        

    def __repr__(self):
        # cabecalho basico com o melhor kappa e os lasers
        linhas = [
            f"<GridResult: Melhor kappa={self.best_kappa:.2f}>",
            f"Lasers (um): {list(np.round(self.best_lambdas, 4))}"
        ]
        
        # detalha as equacoes usadas em cada um dos 4 lasers campeoes
        if self.grid is not None:
            linhas.append("Equações utilizadas:")
            for i, wl in enumerate(self.best_lambdas):
                # pega a referencia de cada gas para esse comprimento de onda
                eqs = [
                    str(self.grid._gas_valid_interval(gas, wl)) 
                    for gas in self.grid.gasesid
                ]
                linhas.append(f"  l_{i+1} ({wl:.4f} um): " + " | ".join(eqs))
        

        return "\n".join(linhas)

if __name__ == '__main__':
    a = np.linspace(0.2, 0.4)
    print("Sample:", a)
    
    c = GridSearch(sample=a, k=4)   

    res = c.run_seach()
    print(res)
    print('N combinacaoes: ', len(c))
    print(res.to_dataframe())