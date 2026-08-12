import numpy as np
from numpy.linalg import norm
from scipy import optimize as opt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from kneed import KneeLocator


from gasopt.gladstone_dale import GasMix

class TikhonovSolver():
    """
    Resolvedor de problemas inversos de refratometria de gases via Regularização de Tikhonov.

    Esta classe formula e resolve a otimização de Tikhonov sujeita a restrições físicas:
    a soma das frações molares deve ser igual a 1, e cada fração molar deve estar no
    intervalo [0, 1]. A seleção do parâmetro de regularização ótimo (alpha) é realizada
    automaticamente através da detecção do ponto de inflexão ("knee") da Curva L.
    """

    def __init__(self, mtx: np.ndarray, sigma_mtx: np.ndarray, b_obs: np.ndarray, sigma_b: np.ndarray):
        """
        Inicializa o resolvedor com as matrizes de refratividade e observações.

        Parâmetros
        ----------
        mtx : np.ndarray
            Matriz de refratividade dos gases componentes (A).
        sigma_mtx : np.ndarray
            Matriz de incertezas/erros dos elementos de mtx.
        b_obs : np.ndarray
            Vetor de refratividades observadas/medidas (b).
        sigma_b : np.ndarray
            Vetor de incertezas/erros associados a b_obs.
        """

        # Melhorando a precisão numérica
        # O sistema é linear: A . x = b
        # Então, o vetor b e as incertezas também são multiplicados por 1e4
        self.mtx = mtx * 1e4
        self.b_obs = b_obs * 1e4

        # Multiplicamos o erro por 1e4 também
        self.sigma_mtx = sigma_mtx * 1e4
        self.sigma_b = sigma_b * 1e4

        # Parâmetros de regularização (grid logarítmico)
        self.alphas = np.logspace(-5, 0, 100)

        # Chute inicial
        self.x0 = np.ones(len(self.b_obs)) / len(self.b_obs)

    def _sqrnorm(self, A: np.ndarray, b: np.ndarray, alpha: float, x: np.ndarray, x0: np.ndarray) -> float:
        """
        Calcula o valor da função objetivo de Tikhonov (norma quadrada do resíduo + regularização).

        Parâmetros
        ----------
        A : np.ndarray
            Matriz do sistema.
        b : np.ndarray
            Vetor de observações.
        alpha : float
            Parâmetro de regularização.
        x : np.ndarray
            Vetor de estado atual (frações molares).
        x0 : np.ndarray
            Vetor a priori / estimativa inicial.

        Retorna
        -------
        float
            Valor do funcional de Tikhonov ||A*x - b||^2 + alpha^2 * ||x - x0||^2.
        """
        return np.sum((A @ x - b) ** 2) + (alpha ** 2) * np.sum((x - x0) ** 2)
    
    def _gradient(self, A: np.ndarray, b: np.ndarray, alpha: float, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente analítico da função objetivo de Tikhonov.

        Parâmetros
        ----------
        A : np.ndarray
            Matriz do sistema.
        b : np.ndarray
            Vetor de observações.
        alpha : float
            Parâmetro de regularização.
        x : np.ndarray
            Vetor de estado atual.
        x0 : np.ndarray
            Vetor a priori.

        Retorna
        -------
        np.ndarray
            Gradiente 2 * A.T @ (A @ x - b) + 2 * alpha^2 * (x - x0).
        """
        return 2 * A.T @ (A @ x - b) + 2 * (alpha ** 2) * (x- x0)

    def minimize_constrain(self, A: np.ndarray, b_noisy: np.ndarray, alpha: float, x0: np.ndarray) -> np.ndarray:
        """
        Resolve a otimização de Tikhonov sujeira a restrições de igualdade e limite.

        Restrições aplicadas:
        - Igualdade: sum(x) = 1.0 (soma das frações molares)
        - Limites de caixa: 0.0 <= x_i <= 1.0

        Parâmetros
        ----------
        A : np.ndarray
            Matriz do sistema linear.
        b_noisy : np.ndarray
            Vetor de dados observados (com ruído).
        alpha : float
            Parâmetro de regularização.
        x0 : np.ndarray
            Ponto inicial para a otimização.

        Retorna
        -------
        np.ndarray
            Vetor de frações molares otimizadas x.
        """
        # Restrição de igualdade: soma das frações molares deve ser igual a 1
        constraints = {'type': 'eq', 
                       'fun': lambda x: np.sum(x) - 1.0}
        
        # Restrições de caixa (limites): 0 <= x_i <= 1
        bounds = opt.Bounds(0, 1.0)
        
        res = opt.minimize(
            fun=lambda x: self._sqrnorm(A, b_noisy, alpha, x, x0), 
            x0=x0, 
            jac=lambda x: self._gradient(A, b_noisy, alpha, x, x0), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-14})

        return res.x

    def get_curve_L(self):
        """
        Calcula os pontos da Curva L variando o parâmetro de regularização alpha.

        Retorna
        -------
        normas_residuo : np.ndarray
            Normas quadradas dos resíduos ||A*x - b||^2 para cada alpha.
        normas_solucao : np.ndarray
            Normas quadradas das soluções ||x||^2 para cada alpha.
        phis : np.ndarray
            Matriz com os vetores de solução (frações molares) para cada alpha.
        """
        # Array para receber os valores
        phis = np.zeros([len(self.alphas), len(self.b_obs)])

        # Listas para armazenar as componentes da Curva L
        normas_residuo = []
        normas_solucao = []

        for i in range(len(phis)):
            phi_calc = self.minimize_constrain(
                A=self.mtx, 
                b_noisy=self.b_obs, 
                alpha=self.alphas[i], 
                x0=self.x0
            )

            phis[i] = phi_calc

            # Cálculo das normas euclidianas quadradas para a Curva L
            res_norm = np.sum((self.mtx @ phi_calc - self.b_obs) ** 2)
            sol_norm = np.sum(phi_calc ** 2)
            
            normas_residuo.append(res_norm)
            normas_solucao.append(sol_norm)

        return np.array(normas_residuo), np.array(normas_solucao), phis
    
    def find_knee(self, x: np.ndarray, y: np.ndarray):
        """
        Localiza o ponto de cotovelo/joelho ("knee") da Curva L.

        Parâmetros
        ----------
        x : np.ndarray
            Coordenadas x da Curva L (normas dos resíduos).
        y : np.ndarray
            Coordenadas y da Curva L (normas das soluções).

        Retorna
        -------
        float ou None
            Valor x correspondente ao ponto de maior curvatura.
        """
        kl = KneeLocator(x, y, curve='concave', direction='decreasing')
        return kl.knee

    def nao_aglomerados(self, x, factor=0.1):
        """
        Retorna os índices dos pontos não aglomerados da Curva L.

        Detecta regiões onde os valores consecutivos variam muito pouco
        (abaixo de um limiar baseado na variação mediana) nas extremidades
        do array.

        Parâmetros
        ----------
        x : np.ndarray
            Array numpy unidimensional de coordenadas.
        factor : float, opcional
            Fração da mediana das diferenças absolutas não-nulas usada como
            limiar para considerar um ponto como "aglomerado". Valores
            menores tornam a detecção mais rigorosa. Padrão: 0.1.

        Retorna
        -------
        np.ndarray
            Índices dos pontos pertencentes a regiões não aglomeradas.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        # Diferença absoluta entre elementos consecutivos
        diffs = np.abs(np.diff(x))

        # Limiar: fração da mediana das diferenças que não são ~zero
        nonzero = diffs[diffs > 0]
        if len(nonzero) == 0:
            # Array inteiramente constante — todos são aglomerados
            return np.arange(n)

        limiar = factor * np.median(nonzero)

        # Encontrar primeiro e último índice com variação acima do limiar
        acima = np.where(diffs > limiar)[0]
        inicio = acima[0]       # primeiro ponto com variação real
        fim = acima[-1] + 1     # +1 porque diff tem tamanho n-1

        # Índices aglomerados: antes do início e depois do fim
        nao_aglomerados = np.arange(inicio, fim+1)

        return nao_aglomerados
    
    def find_max_curvature_spline(self, x: np.ndarray, y: np.ndarray) -> int:
        """
        Ajusta uma spline à curva L e calcula a curvatura κ analiticamente.
        """
        # Normalizar para escalas comparáveis
        x_n = (x - x.min()) / (x.max() - x.min())
        y_n = (y - y.min()) / (y.max() - y.min())

        # Parametrizar por comprimento de arco acumulado
        t = np.zeros(len(x_n))
        for i in range(1, len(x_n)):
            t[i] = t[i-1] + np.sqrt((x_n[i]-x_n[i-1])**2 + (y_n[i]-y_n[i-1])**2)

        t /= t[-1]  # normalizar para [0, 1]
        
        # t precisa ser estritamente crescente para UnivariateSpline
        # entao eu crio um lista de booleanos
        # e cada vez que a diferença é menor que 1e-12 (menor que um milionesimo)
        # eu considero o ponto como repetido
        # e eu removo os pontos repetidos
        valid = np.ones(len(t), dtype=bool)
        valid[1:] = np.diff(t) > 1e-12

        t = t[valid]
        x_n = x_n[valid]
        y_n = y_n[valid]

        # Ajustar splines separadas para x(t) e y(t)
        spline_x = UnivariateSpline(t, x_n, s=0)
        spline_y = UnivariateSpline(t, y_n, s=0)
        
        # Derivadas
        dx = spline_x.derivative()(t)
        ddx = spline_x.derivative(n=2)(t)
        dy = spline_y.derivative()(t)
        ddy = spline_y.derivative(n=2)(t)
        
        # Guardar os índices originais dos pontos válidos
        indices_originais = np.where(valid)[0]

        # Curvatura κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        curve = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        # Mapear o índice de curvatura máxima de volta para o vetor original
        return int(indices_originais[np.argmax(curve)]), curve
    
    def solve(self, log: bool = False, factor=0.1, method: str = 'spline'):
        """
        Executa o fluxo completo de solução por Regularização de Tikhonov.

        Calcula a Curva L, filtra regiões aglomeradas, identifica o alpha
        ótimo e retorna o objeto TikSolution.

        Parâmetros
        ----------
        log : bool, opcional
            Se True, aplica log10 às normas antes da detecção. Padrão: False.
        factor : float, opcional
            Fator de filtragem de pontos aglomerados. Padrão: 0.1.
        method : str, opcional
            Método para detecção do ponto ótimo da Curva L. Opções:
            - 'spline': Curvatura geométrica κ via derivadas de spline (recomendado).
            - 'kneed' : Detecção de "knee" via biblioteca Kneed (KneeLocator).
            Padrão: 'spline'.

        Retorna
        -------
        TikSolution
            Objeto contendo a solução ótima, alpha ótimo e históricos de normas.

        Raises
        ------
        ValueError
            Se o método especificado não for reconhecido.
        """
        # Métodos disponíveis para detecção do ponto ótimo
        valid_methods = ('kneed', 'spline')
        if method not in valid_methods:
            raise ValueError(
                f"Método '{method}' não reconhecido. Use: {valid_methods}"
            )

        # 1. Calcular a Curva L completa
        x, y, phis = self.get_curve_L()

        # 2. Aplicar escala logarítmica (recomendado para Curva L)
        if log:
            x = np.log10(x)
            y = np.log10(y)

        # 3. Filtrar pontos aglomerados nas extremidades
        keep = self.nao_aglomerados(x, factor=factor)
        x = x[keep]
        y = y[keep]
        phis = phis[keep]
        good_alphas = self.alphas[keep]

        # 4. Encontrar o ponto ótimo da Curva L pelo método escolhido
        if method == 'kneed':
            # Kneedle: retorna o valor x correspondente ao knee
            knee = self.find_knee(x, y)
            knee_index = np.where(x == knee)[0][0]

        elif method == 'spline':
            # Curvatura geométrica via spline: retorna diretamente o índice
            knee_index, curve_vals = self.find_max_curvature_spline(x, y)

        # 5. Extrair a solução ótima
        optimal_alpha = good_alphas[knee_index]
        solution = phis[knee_index]

        return TikSolution(x, y, good_alphas, optimal_alpha, solution, knee_index, curve_vals)

class TikSolution():
    """
    Encapsula o resultado final da solução por Regularização de Tikhonov.

    Armazena as normas dos resíduos, normas das soluções, o parâmetro alpha ótimo
    encontrado e o vetor de frações molares ótimas.
    """

    def __init__(self, res_norms: np.ndarray, sol_norms: np.ndarray, alphas: np.ndarray, opt_alpha, opt_phi: np.ndarray, opt_index: int, curv: np.ndarray):
        """
        Inicializa o objeto com os resultados da otimização.

        Parâmetros
        ----------
        res_norms : np.ndarray
            Normas dos resíduos correspondentes aos alphas filtrados.
        sol_norms : np.ndarray
            Normas das soluções correspondentes aos alphas filtrados.
        alphas : np.ndarray
            Vetor completo de parâmetros de regularização testados.
        opt_alpha : float ou np.ndarray
            Parâmetro de regularização ótimo selecionado.
        opt_phi : np.ndarray
            Vetor de frações molares ótimas correspondente ao opt_alpha.
        opt_index: int
        """
        self.residual_norm = res_norms
        self.solution_norm = sol_norms
        self.curvature = np.abs(curv)
        self.alphas = alphas
        self.opt_alpha = opt_alpha
        self.opt_phi = opt_phi
        self.opt_index = opt_index

        # encontrando onde a curvatura é a maior
        #optIdx = np.argmax(np.abs(self.curvature)) 
        #self.opt_alpha = self.alphas[optIdx]
    
    def __str__(self) -> str:
        """
        Retorna uma representação formatada em string da solução.
        """
        return (
            f"Optimal Alpha: {self.opt_alpha}\n"
            f"Optimal Phi: {self.opt_phi}\n"
            f"Residual Norm: {self.residual_norm}\n"
            f"Solution Norm: {self.solution_norm}\n"
        )