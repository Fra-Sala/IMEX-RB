import sys
import os
# Go two levels up from current file (i.e., from notebooks/ to project/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

# Apply publication style
import utils.mpl_pubstyle  # noqa: F401


def main():
    """Plot convergence of BE and IMEX-RB from saved results Burgers2D."""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Paths
    problem_name = "Burgers2D"
    results_path = os.path.join(
        base_dir, os.pardir, os.pardir, 'results',
        problem_name, 'convergence', 'Test2', 'results.npz'
    )
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    data = np.load(results_path, allow_pickle=True)
    errors_l2 = data['errors_l2'].item() if data['errors_l2'].dtype == object else data['errors_l2']
    Nt_values = data['Nt_values']

    dts = 1.0 / np.array(Nt_values)
    methods = ['BE', 'IMEX-RB']
    comp_labels = ['u_x', 'u_y']

    for comp in range(errors_l2[methods[0]].shape[0]):
        plt.figure(figsize=(6, 4))
        for m in methods:
            errs = errors_l2[m][comp]
            plt.loglog(dts, errs, marker='o', label=m)
        # Reference line
        plt.loglog(dts, dts, label=r"$\mathcal{O}(\Delta t)$",
                   color="k", linestyle='--')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel(fr'$\|e(t)\|_{{l^2,{comp_labels[comp]}}}$')
        # plt.title(f'Convergence for {comp_labels[comp]}')
        plt.legend()
        plt.tight_layout()
        out_file = os.path.join(
            plots_dir,
            f'{problem_name}_convergence_{comp_labels[comp]}.pdf'
        )
        plt.savefig(out_file)
        plt.close()


if __name__ == '__main__':
    main()
