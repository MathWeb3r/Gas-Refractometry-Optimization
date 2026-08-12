import matplotlib.pyplot as plt
import numpy as np

def plot_curva_L(alfas, normas_residuo, normas_solucao, peak_idx = None, plot_alfa=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Desenha a linha de tendência da Curva L
    ax.plot(normas_residuo, normas_solucao, 'b-', linewidth=2, label='Caminho de Tikhonov')
    
    # Plota os pontos coloridos variando de acordo com o log10(alfa)
    sc = ax.scatter(normas_residuo, normas_solucao, c=np.log10(alfas), 
                    cmap='jet', s=35, zorder=5)
    
    # Configurações de eixos e labels em escala logarítmica
    ax.set_title("Curva L Dinâmica", fontsize=12)
    ax.set_xlabel(r"Norma do Resíduo: $\log_{10} ||A\Phi - b_{noise}||_2^2$", fontsize=10)
    ax.set_ylabel(r"Norma da Solução: $\log_{10} ||\Phi||_2^2$", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Barra de cores lateral para identificar visualmente o Alfa ideal
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Parâmetro de Regularização $\log_{10}(\alpha)$")

    if peak_idx != None:
        ax.axvline(normas_residuo[peak_idx], color='k', linestyle='--')

    if plot_alfa:
        for idx, a in enumerate(alfas):
            a = np.log10(a)
            ax.text(normas_residuo[idx]*1.01, normas_solucao[idx]*1.01, 
                    s=f'$\\text{{log}}\\alpha={a:.1f}$')
    
    ax.legend()
    return fig, ax

def plot_curvatura(alfas, curv):
    fig, ax = plt.subplots(figsize=(7, 5))
        
    ax.plot(np.log10(alfas), np.abs(curv), 'r-', linewidth=2, label=r'$|\kappa(\alpha)|$')
    
    alfa_otimo = alfas[np.argmax(curv)]
    ax.axvline(x=np.log10(alfa_otimo), color='gold', linestyle='--', linewidth=2, 
               label=f'Vértice Ótimo: $\\log_{10}(\\alpha) = {np.log10(alfa_otimo):.2f}$')
    
    ax.set_title("Análise de Curvatura Analítica", fontsize=11)
    ax.set_xlabel(r"Parâmetro de Regularização $\log_{10}(\alpha)$")
    ax.set_ylabel(r"Curvatura Absoluta $|\kappa|$")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')

def add_text_panel(fig, lambdas = None, cond_num = None, erro = None):
    # Redimensiona a figura
    fig.set_figwidth(12)

    # Restringe a plotagem principal até 75% da largura
    fig.subplots_adjust(right=0.75)  

    # Cria um eixo para o painel lateral
    ax_side = fig.add_axes([0.77, 0.15, 0.18, 0.70])
    ax_side.axis('off')
    
    text_lines = []
    if cond_num is not None or erro is not None:
        text_lines.append(r"$\mathbf{Par\hat{a}metros\ do\ Sistema}$")
        text_lines.append("─" * 20)
        if cond_num is not None:
            if isinstance(cond_num, float):
                val_str = f"{cond_num:.2e}" if cond_num > 1e4 else f"{cond_num:.4f}"
                text_lines.append(rf"$\kappa(A) = {val_str}$")
            else:
                text_lines.append(rf"$\kappa(A) = {cond_num}$")
        if erro is not None:
            if isinstance(erro, float):
                erro_pct = erro * 100
                if erro_pct < 0.01 and erro_pct > 0:
                    text_lines.append(rf"$\text{{Erro: }} {erro_pct:.4f}\%$")
                else:
                    text_lines.append(rf"$\text{{Erro: }} {erro_pct:.2f}\%$")
            else:
                text_lines.append(rf"$\text{{Erro: }} {erro}$")
        text_lines.append("")
    
    if lambdas is not None:
        text_lines.append(r"$\mathbf{Comprimentos\ de\ Onda\ (\lambda):}$")
        text_lines.append("─" * 20)
        for idx, lam in enumerate(lambdas, start=1):
            if isinstance(lam, (float, int)):
                unit = r"\text{ nm}" if lam > 100 else r"\text{ }\mu\text{m}"
                text_lines.append(rf"$\lambda_{{{idx}}} = {lam}{unit}$")
            else:
                text_lines.append(rf"$\lambda_{{{idx}}} = {lam}$")
    
    side_text = "\n".join(text_lines)
    ax_side.text(0.05, 0.95, side_text, transform=ax_side.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9))