import sys
import os
# Go two levels up from current file (i.e., from notebooks/ to project/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
import numpy as np
import matplotlib.pyplot as plt

import utils.mpl_pubstyle  # noqa: F401


def main():
    """Plot energy norm errors versus time for each method.
    We are plotting the absolute error."""

    base_dir = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(
        base_dir, os.pardir, os.pardir, 'results',
        'Burgers2D', 'energynorm', 'Test2', 'results.npz'
    )
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    data = np.load(results_path, allow_pickle=True)
    errors_energy = data['errors_energy'].item()
    Nt = data['Nt'].item()
    tvec = np.linspace(0, 1, Nt + 1)  # assuming time interval is [0, 1]

    methods = ['BE', 'IMEX-RB', 'FE']
    comp_labels = ['u_x', 'u_y']
    linestyles = ['-', '--']

    plt.figure(figsize=(8, 5))
    for i, comp in enumerate(comp_labels):
        for m in methods:
            e = errors_energy[m][i]
            plt.semilogy(tvec[1:], e, linestyle=linestyles[i], label=f'{m}, {comp}')

    plt.xlabel('Time $t$')
    plt.ylabel(r'$\|e(t)\|_{\mathcal{E}}^2$')
    plt.ylim(top=1e10)
    plt.title('Energy Norm of the Error Over Time')
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(plots_dir, 'Burgers2D_energynorm_vs_time.pdf')
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    main()
