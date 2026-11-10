import numpy as np
import emcee as mc
import matplotlib.pyplot as plt

class MCMCInference:
    def __init__(self, mtx, sigma_mtx=None, phi_real=None, erro=None, b_obs=None, sigma=None, sigma_b=None,
                 x0=None, sigma_x0=None):
        """
        Inicializa a classe de inferência Bayesiana MCMC.
        
        Args:
            mtx (numpy.ndarray): Matriz de refratividades (M comprimentos de onda x N gases).
            sigma_mtx (numpy.ndarray, opcional): Incerteza associada aos elementos da matriz mtx.
            phi_real (numpy.ndarray, opcional): Vetor phi real (frações molares). 
                Usado para simular b_obs e sigma caso não sejam fornecidos diretamente.
            erro (float, opcional): Erro percentual para simulação de b_obs (ex: 0.001 para 0.1%).
            b_obs (numpy.ndarray, opcional): Vetor de refratividade observado (M elementos).
            sigma (numpy.ndarray ou float, opcional): Incerteza instrumental associada às observações b_obs.
            sigma_b (numpy.ndarray ou float, opcional): Sinônimo de sigma para compatibilidade com o pacote.
            x0 (numpy.ndarray ou list, opcional): Estimativa prévia da composição (N gases).
                - Com sigma_x0: é o centro da priori gaussiana (priori informativa).
                - Sem sigma_x0: a priori continua uniforme e x0 serve apenas
                  como centro de inicialização dos walkers.
            sigma_x0 (float, list ou numpy.ndarray, opcional): Largura da priori gaussiana em torno de x0.
                Pode ser um valor único (mesma largura para todos os gases) ou um vetor de tamanho N.
                Se não for fornecido, a priori é uniforme no simplex (não-informativa).
                Exige que x0 seja fornecido.
        """
        self.mtx = mtx
        self.sigma_mtx = sigma_mtx
        
        # o numero de dimensões é igual 
        # ao numero de gases 
        # menos 1 (dependencia linear)
        self.n_gases = mtx.shape[1]
        self.ndim = self.n_gases - 1
        
        self.b_obs = b_obs

        # compatibilidade 
        # sigma é o erro instrumental de b
        # em outras classes desse pacote, é chamado sigma_b
        if sigma_b is not None and sigma is not None:
            raise ValueError("É necessário fornecer apenas sigma ou sigma_b, não ambos.")
        elif sigma_b is not None:
            self.sigma = sigma_b
        else:
            self.sigma = sigma

        if b_obs is not None and self.sigma is not None:
            self.b_obs = b_obs
        elif phi_real is not None and erro is not None:
            # a refratividade real
            b_real = np.dot(mtx, phi_real)
            
            # adicionando um ruído gaussiano baseado no erro percentual
            self.sigma = erro * np.abs(b_real)
            self.b_obs = b_real + np.random.normal(loc=0, scale=self.sigma, size=b_real.shape)
        else:
            raise ValueError("É necessário fornecer b_obs e sigma (ou sigma_b), ou phi_real e erro para simulação.")

        # configuração da priori
        # (uniforme ou gaussiana em torno de x0)
        self._setup_prior(x0, sigma_x0)

        self.sampler = None

    def _setup_prior(self, x0, sigma_x0):
        """
        Configura a informação a priori p(phi).

        Define os atributos self.x0, self.sigma_x0 e self.prior_type:
            - sigma_x0 = None -> prior_type = 'uniform'
              (apenas os limites físicos do simplex)
            - sigma_x0 dado   -> prior_type = 'informative'
              (gaussiana centrada em x0, truncada no simplex)

        Args:
            x0 (array_like ou None): Estimativa prévia da composição (N gases).
            sigma_x0 (float, array_like ou None): Largura da priori. Valor único
                (aplicado a todos os gases) ou vetor de tamanho N.

        Raises:
            ValueError: Se os tamanhos forem incompatíveis, se sigma_x0 tiver
                valores não positivos, ou se sigma_x0 for dado sem x0.
        """
        # x0 é opcional: mesmo com priori uniforme
        # ele pode servir para inicializar os walkers
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.n_gases,):
                raise ValueError(f"x0 deve ter tamanho {self.n_gases} (número de gases).")
        self.x0 = x0

        # sem sigma_x0 -> não há informação a priori
        # a priori é uniforme no simplex (ln(1) = 0)
        if sigma_x0 is None:
            self.sigma_x0 = None
            self.prior_type = 'uniform'
            return

        # a priori gaussiana precisa de um centro
        if x0 is None:
            raise ValueError("sigma_x0 exige que x0 seja fornecido (centro da priori).")

        # np.broadcast_to transforma um valor único em
        # um vetor de tamanho N, e mantém vetores de tamanho N.
        # Qualquer outro tamanho gera erro
        try:
            sigma_x0 = np.broadcast_to(np.asarray(sigma_x0, dtype=float), (self.n_gases,)).copy()
        except ValueError:
            raise ValueError(f"sigma_x0 deve ser um valor único ou ter tamanho {self.n_gases}.")

        # a largura da gaussiana deve ser positiva
        # (sigma_x0 = 0 geraria divisão por zero no log_prior)
        if np.any(sigma_x0 <= 0):
            raise ValueError("Todos os valores de sigma_x0 devem ser estritamente positivos.")

        self.sigma_x0 = sigma_x0
        self.prior_type = 'informative'

    def log_prior(self, x):
        """
        Calcula o logaritmo da probabilidade a priori p(phi).
        
        Verifica o sentido físico do vetor (pertencimento ao simplex: 0 <= phi_i <= 1 e soma = 1.0)
        e adiciona a penalidade gaussiana centrada em x0 caso a priori informativa esteja ativa.

        Args:
            x (numpy.ndarray): Primeiras (N - 1) frações molares independentes.

        Returns:
            float: log(p(phi)) ou -inf se violar as restrições físicas.
        """
        #
        # O prior é p(phi)
        # É a probabilidade do meu vetor de solução ser real antes de olhar os dados b.
        # Aqui eu verifico o sentido físico desse vetor:
        # return -inf <--> estado fisicamente impossível (pois log(0) = -inf)
        # return 0.0  <--> estado possível com densidade uniforme (pois log(1) = 0)
        #
        
        # Restrição física para os primeiros (N - 1) no intervalo de [0, 1]
        if np.any(x < 0.0) or np.any(x > 1.0):
            return -np.inf

        # Restrição física do último elemento dependente: 0 <= lastx <= 1
        lastx = 1.0 - np.sum(x)
        if lastx < 0.0 or lastx > 1.0:
            return -np.inf

        # Avaliação da densidade de prob. a priori
        # Se for não-informativa (sem sigma_x0), todos os estados no interior
        # do simplex são igualmente prováveis a priori (ln(1) = 0.0)
        if self.prior_type == 'uniform' or self.x0 is None or self.sigma_x0 is None:
            return 0.0
        
        # Se for informativa (com sigma_x0), aplica a densidade gaussiana centrada em x0
        # log p(phi) ~ -0.5 * sum((phi_j - x0_j)^2 / sigma_x0_j^2)
        elif self.prior_type == 'informative':
            x_completo = np.append(x, lastx)
            diff = x_completo - self.x0
            log_p = -0.5 * np.sum((diff / self.sigma_x0) ** 2)
            return log_p

    def log_likelihood(self, x):
        """
        likelihood é P(b | phi)
        mede o quanto b produz o resultado o phi
        Supondo que a distribuição dos erros é gaussiana
        a função de custo é o Chi² do fitting
        Chi²(phi) = Sum (b_i - (A . phi))²/sigma²
        
         P(b|phi) ~ exp(-0.5 * Chi²)
         log P(b|phi) ~ -0.5 * Chi²
        """
        
        # reconstruindo o vetor completo
        lastx = 1 - sum(x)
        x_completo = np.append(x, lastx)
        x_completo = np.array(x_completo)

        # propagação de incerteza
        # a incerteza total é a soma das incertezas
        # instrumentais e da incerteza dos gases
        #
        # sigma2_total: incerteza total quadrada
        #
        # contribuição instrumental
        sigma2_total = self.sigma**2

        # add contribuição da incerteza dos gases
        if self.sigma_mtx is not None:
            sigma2_total += np.sum((self.sigma_mtx ** 2) * (x_completo ** 2), axis=1)

        # A refratividade real
        b_teorico = np.dot(self.mtx, x_completo)

        # Função de custo = chi
        chi2 = (self.b_obs - b_teorico)**2
        chi2 = chi2/sigma2_total
        chi2 = np.sum(chi2)    

        # log likelihood
        likelihood = -0.5*chi2

        return likelihood

    def log_prob(self, x):
        """
        a probabilidade p(phi|b) 
        é a prob de phi ser a solução correta, dado b
        p(phi|b) ~ p(b|phi) * p(phi)  
        log[p(phi|b)] ~ log[p(b|phi)] + log[p(phi)] 
        """
        # log[p(phi)]
        prior = self.log_prior(x)

        # log[p(b|phi)]
        likelihood = self.log_likelihood(x)

        # log[p(phi|b)]
        prob = prior + likelihood
        return prob

    def sequencia(self, spread=1e-3):
        """
        Gera uma posição aleatória estritamente válida para a inicialização de um walker.

        Se houver uma estimativa prévia x0 fornecida, inicializa o walker em uma pequena
        vizinhança gaussiana ao redor de x0 (com desvio padrão dado por spread), garantindo
        que permaneça no interior estrito do simplex (todas as componentes > 0 e soma < 1.0).
        Caso contrário, amostra uniformemente no simplex através da distribuição de Dirichlet (alpha=1).

        Args:
            spread (float): Dispersão gaussiana em torno de x0 (padrão: 1e-3).

        Returns:
            numpy.ndarray: Vetor de tamanho self.ndim com as frações molares independentes.
        """
        # Caso 1: Se x0 foi fornecido, concentra os walkers em uma bola ao redor de x0
        # Isso reduz drasticamente o tempo de burn-in (Foreman-Mackey et al., 2013)
        if self.x0 is not None:
            for _ in range(1000):
                ruido = np.random.normal(loc=0.0, scale=spread, size=self.ndim)
                cand = self.x0[:self.ndim] + ruido
                last_cand = 1.0 - np.sum(cand)
                # Verifica se todas as frações permanecem estritamente positivas
                if np.all(cand > 0.0) and (last_cand > 0.0):
                    return cand

        # Caso 2: Priori uniforme / sem x0 -> amostra uniformemente pelo simplex via Dirichlet
        fractions = np.random.dirichlet(np.ones(self.ndim + 1))
        return fractions[:self.ndim]

    def gerar_p0(self, n_walkers, spread=1e-3):
        """
        Gera uma matriz de posições iniciais (n_walkers, ndim) para os walkers,
        garantindo consistência com a física do problema (interior estrito do simplex).
        
        Args:
            n_walkers (int): Número de walkers no ensemble.
            spread (float): Dispersão caso x0 esteja definido.

        Returns:
            numpy.ndarray: Matriz (n_walkers, ndim) de posições iniciais válidas.
        """
        p0 = np.zeros((n_walkers, self.ndim))
        for i in range(n_walkers):
            p0[i] = self.sequencia(spread=spread)

        return p0

    def get_sampler(self, n_walkers=32):
        """
        Cria e retorna o EnsembleSampler do emcee.
        """
        # criando o sampler
        sampler = mc.EnsembleSampler(
            nwalkers=n_walkers, 
            ndim=self.ndim, 
            log_prob_fn=self.log_prob
        )

        return sampler
        
    def run_mcmc(self, n_walkers: int = 32, n_steps: int = 7500, 
                 p0: np.ndarray = None, progress: bool = False, prior_type: str = None):
        """ 
        Configura o sampler, gera o ponto inicial (p0) e executa o MCMC.

        Args:
            n_walkers (int): Número de walkers (padrão: 32).
            n_steps (int): Número de passos por walker (padrão: 7500).
            p0 (numpy.ndarray, opcional): Posições iniciais dos walkers (formato: n_walkers, ndim).
                Se não for fornecido, será gerado automaticamente por gerar_p0().
            progress (bool): Se True, exibe barra de progresso no terminal.
            prior_type (str, opcional): Permite alternar a priori nesta execução ('uniform' ou 'informative').
                Se None, preserva o tipo configurado na inicialização da classe.

        Returns:
            emcee.EnsembleSampler: O objeto sampler após a execução.
        """
        # Permite alterar ou forçar a priori pontualmente
        if prior_type is not None:
            self.prior_type = prior_type

        # É preciso definir um ponto de partida para cada walker
        # Se o usuário não forneceu p0 customizado, gera automaticamente
        if p0 is None:
            p0 = self.gerar_p0(n_walkers=n_walkers)
        else:
            p0 = np.asarray(p0, dtype=float)
            if p0.shape != (n_walkers, self.ndim):
                raise ValueError(f"p0 deve ter formato ({n_walkers}, {self.ndim}), recebido {p0.shape}.")
        
        sampler = self.get_sampler(n_walkers=n_walkers)
        self.sampler = sampler

        sampler.run_mcmc(
            initial_state=p0,
            nsteps=n_steps,
            progress=progress
        )
        
        return sampler
    
    def get_results(self, sampler = None, burn_in=300):
        """
        Extrai as cadeias de resultado a partir de um sampler percorrido.
        """

        if sampler == None and self.sampler == None:
            raise ValueError("No sampler provided. Run run_mcmc() or get_sampler() first.")
        
        elif sampler == None and self.sampler != None:
            sampler = self.sampler

        # cadeia walker a walker
        # espero veficar a convergencia
        chain = sampler.get_chain()
        
        # cadeia "achatada"
        # espero veficar a distribuição posterior
        chainflat = sampler.get_chain(discard=burn_in, flat=True)

        # Recupera o componente de referência x_k = 1 - sum(x_i)
        # pela dependência linear do fechamento do simplex.
        # Na cadeia completa (steps, walkers, ndim)
        lastx_chain = 1.0 - np.sum(chain, axis=-1, keepdims=True)
        chain = np.concatenate([chain, lastx_chain], axis=-1)
        
        # Na cadeia achatada (n_samples, ndim)
        lastx_flat = 1.0 - np.sum(chainflat, axis=-1, keepdims=True)
        chainflat = np.concatenate([chainflat, lastx_flat], axis=-1)
        
        return chain, chainflat

    @staticmethod
    def mean_interval(chainflat, confidence=0.95):
        """
        Cálculo da média e intervalo de credibilidade (bayesiano) para cada parâmetro.
        
        Args:
            chainflat (np.ndarray): Cadeia no formato achatado (n_samples, n_params).
            confidence (float): Nível de confiança do intervalo (padrão: 0.95 para 95%).
            
        Returns:
            tuple: (mean_vector, intervals_matrix)
                - mean_vector: Vetor 1D com a média de cada parâmetro.
                - intervals_matrix: Matriz 2D (n_params, 2) com [lower_bound, upper_bound] para cada parâmetro.
        """
        n_params = chainflat.shape[1]
        
        # 1. Calcular a média para cada parâmetro
        mean_vector = np.mean(chainflat, axis=0)
        
        # 2. Calcular os limites do intervalo de credibilidade
        # Para um intervalo de confiança de 95%, queremos os quantis 0.025 e 0.975
        lower_bound = (1.0 - confidence) / 2.0
        upper_bound = 1.0 - lower_bound
        
        intervals_matrix = np.zeros((n_params, 2))
        
        for i in range(n_params):
            param_chain = chainflat[:, i]
            
            # Calcula os quantis (limites do intervalo de credibilidade)
            intervals_matrix[i, 0] = np.quantile(param_chain, lower_bound)
            intervals_matrix[i, 1] = np.quantile(param_chain, upper_bound)
        
        return mean_vector, intervals_matrix

    def get_map(self, burn_in=300):
        """
        Retorna o estimador MAP (Maximum A Posteriori) - o ponto com maior log_prob na cadeia.
        
        Para distribuições assimétricas ou truncadas na fronteira física (phi >= 0),
        o MAP representa o valor mais provável (pico da distribuição / moda), sendo imune
        ao viés de arrasto que afeta a média devido à cauda estendida.
        
        Args:
            burn_in (int): Passos iniciais a descartar.
            
        Returns:
            numpy.ndarray: Vetor 1D com a fração molar de cada gás no ponto de maior probabilidade.
        """
        if self.sampler is None:
            raise ValueError("MCMC ainda não foi executado. Execute run_mcmc() primeiro.")
            
        # log_prob achatado após descartar o burn-in
        log_probs = self.sampler.get_log_prob(discard=burn_in, flat=True)
        best_idx = np.argmax(log_probs)
        
        _, chainflat = self.get_results(burn_in=burn_in)
        return chainflat[best_idx]

    def solucao(self, burn_in=300):
        """
        Retorna a solução pontual da posterior e o intervalo de credibilidade de 95%.
        
        Retorna tanto a média quanto o MAP (ponto mais provável) e a mediana, permitindo
        comparar estimadores quando a distribuição é assimétrica.

        Args:
            burn_in (int): Quantidade de passos iniciais a descartar.
            
        Returns:
            tuple: (media, intervalo) - mantido para compatibilidade,
                   onde media é o vetor de médias e intervalo é matriz (N, 2).
        """
        chain, chainflat = self.get_results(burn_in=burn_in)
        media, intervalo = self.mean_interval(chainflat)
        return media, intervalo

    def solucao_completa(self, burn_in=300, confidence=0.95):
        """
        Retorna um dicionário com todos os estimadores pontuais e intervalos:
        - 'mean': Média posterior (MMSE)
        - 'map': Moda / valor mais provável (Maximum A Posteriori)
        - 'median': Mediana posterior (quantil 50%)
        - 'std': Desvio padrão posterior
        - 'ci': Matriz (N, 2) de intervalo de credibilidade
        """
        chain, chainflat = self.get_results(burn_in=burn_in)
        mean_vec, intervals = self.mean_interval(chainflat, confidence=confidence)
        median_vec = np.median(chainflat, axis=0)
        std_vec = np.std(chainflat, axis=0)
        map_vec = self.get_map(burn_in=burn_in)
        
        return {
            'mean': mean_vec,
            'map': map_vec,
            'median': median_vec,
            'std': std_vec,
            'ci': intervals
        }

def plot_trace(chain, burn_in=300, labels=None):
    """
    Plota o trace dos walkers.
    
    Args:
        chain (numpy.ndarray): Cadeia no formato (walkers, passos, dimensões).
        burn_in (int): Número de passos a serem marcados como burn-in.
        labels (list, opcional): Lista com os nomes dos parâmetros.
    """
    ndim = chain.shape[-1]
    if labels is None:
        labels = [f"Param {i+1}" for i in range(ndim)]
        
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), sharex=True)
    if ndim == 1:
        axes = [axes] # Garante que seja iterável para 1 dimensão

    for i in range(ndim):
        axes[i].plot(chain[:, :, i], color='black', alpha=0.3)
        axes[i].axvline(burn_in, color='red', linestyle='--', label='Burn in')
        axes[i].set_ylabel('Valor', fontsize=11)
        axes[i].set_title(f'Trace de {labels[i]}', fontsize=12)
        axes[i].grid(alpha=0.3)
        
    axes[ndim-1].set_xlabel('Passos', fontsize=12)    
    fig.tight_layout()
    fig.canvas.draw()
    return fig.canvas.buffer_rgba()

def plot_distribution(chainflat, x_real=None, x_map=None, titulo="Distribuição Posterior", labels=None):
    """
    Plota a distribuição posterior a partir da cadeia achatada.
    
    Args:
        chainflat (numpy.ndarray): Cadeia no formato achatado (passos, dimensões).
        x_real (list ou numpy.ndarray, opcional): Valores reais para plotar a linha de comparação.
        x_map (list ou numpy.ndarray, opcional): Estimador MAP (Moda/pico) para plotar a linha do valor mais provável.
        titulo (str): Título da figura.
        labels (list, opcional): Lista com os nomes dos parâmetros.
    """
    ndim = chainflat.shape[-1]
    if labels is None:
        labels = [f"Param {i+1}" for i in range(ndim)]
        
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), sharex=True)
    if ndim == 1:
        axes = [axes] # Garante que seja iterável
        
    fig.suptitle(titulo, fontsize=16, y=0.96)
    
    for i in range(ndim):
        data = chainflat[:, i]    
        mean = np.mean(data)
        axes[i].hist(data, bins=60, color='gray', alpha=0.6, density=True)
        axes[i].axvline(mean, color='red', linestyle='--', linewidth=2, label='Média Posterior')
        
        if x_map is not None and len(x_map) > i:
            axes[i].axvline(x_map[i], color='green', linestyle='-', linewidth=2, label='MAP (Moda/Pico)')

        if x_real is not None and len(x_real) > i:
            axes[i].axvline(x_real[i], color='blue', linestyle='--', linewidth=2, label='Real')
            
        axes[i].axvspan(mean-np.std(data), mean+np.std(data), alpha=0.2, color='gray', label='1 desvio padrão')
        axes[i].set_ylabel('Densidade', fontsize=11)
        axes[i].set_title(f'Posterior de {labels[i]}', fontsize=12)
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.3)

    axes[ndim-1].set_xlabel('Valor', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90) # Ajuste sutil para o suptitle não sobrepor o tracejado
    fig.canvas.draw()
    return fig
