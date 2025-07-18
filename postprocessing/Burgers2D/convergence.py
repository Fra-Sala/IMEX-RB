import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re, glob
# Set to LateX
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble":
    r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}"
})


# # Plot convergence for IMEX-RB on 2D nonlinear problem
# We study convergence on the 2D viscous Burgers equation
def main():
    """
    Process and plot convergence test for 2D Burgers' problem.
    """
    # Locate parent folder
    try:
        # in a .py file
        base_dir = Path(__file__).resolve().parents[2]
    except NameError:
        # fallback (in a notebook)
        base_dir = Path().resolve().parents[1]
    problem_name = "Burgers2D"
    test_name = "convergence"
    results_base = os.path.join(base_dir, '__RESULTS', problem_name)
    convergence_dirs = glob.glob(os.path.join(results_base, f'{test_name}*'))
    latest_num = max([int(re.search(rf'{test_name}(\d+)',
                                    os.path.basename(d)).group(1))
                     for d in convergence_dirs])
    results_path = os.path.join(results_base, f'{test_name}{latest_num}',
                                'results.npz')

    plots_dir = os.path.join(base_dir, 'postprocessing/Burgers2D/plots')
    os.makedirs(plots_dir, exist_ok=True)

    data = np.load(results_path, allow_pickle=True)
    errors_l2 = data['errors_l2'].item()
    Nt_values = data['Nt_values']
    epsilon_values = data['epsilon_values']
    NtFE = 400
    dts = 1.0 / Nt_values

    all_markers = ['v', '*', 'X', 'd', 'o', 'p', 's', 'h']
    all_colors = [
        'purple', 'blueviolet', 'magenta', 'hotpink',
        'red', 'maroon', 'teal', 'orange'
    ]
    larger_fontsize = 21
    smaller_fontsize = 19

    fig, ax = plt.subplots(figsize=(3, 4.8))  # Create figure and axes

    legend_elements = []
    legend_labels = []

    # Plot BE results
    errs_be = errors_l2['BE']
    line_be = ax.loglog(dts, errs_be, marker='s', markersize=6,
                        color='green', linestyle='--')

    legend_elements.append(line_be[0])
    legend_labels.append('BE')

    # Plot reference line for O(dt)
    ax.loglog(
        dts,
        [dt / 100.0 for dt in dts],
        color='k',
        linestyle='--',
    )

    # Plot vertical line for FE timestep
    ax.axvline(1 / NtFE, color='k', linestyle='-.')

    # Plot IMEX-RB results for each epsilon
    for ie, eps in enumerate(epsilon_values):
        errs_imex = errors_l2['IMEX-RB'][:, ie]
        expnt = int(np.log10(eps))
        label_imex = rf'IMEX-RB, $\varepsilon = 10^{{{expnt}}}$'

        line_imex = ax.loglog(
            dts,
            errs_imex,
            marker=all_markers[ie],
            markersize=6,
            color=all_colors[ie],
            linestyle='-'
        )

        legend_elements.append(line_imex[0])
        legend_labels.append(label_imex)

    # Add text annotation for FE timestep (after all plotting to get y-limits)
    ax.text(
        1 / NtFE * 1.1,
        ax.get_ylim()[1] * 0.6,
        r'$\Delta t_{{\mathrm{{FE}}}}$',
        rotation=0,
        verticalalignment='center',
        color='k',
        fontsize=smaller_fontsize
    )

    # Reference slope triangle
    x0, y0 = 0.02, 0.0001
    dx = 0.02
    dy = y0 * ((x0+dx) / x0) - y0
    ax.plot([x0, x0 + dx], [y0, y0], 'k-', linewidth=0.75)
    ax.plot([x0 + dx, x0 + dx], [y0, y0 + dy], 'k-', linewidth=0.75)
    ax.plot([x0 + dx, x0], [y0 + dy, y0], 'k-', linewidth=0.75)
    ax.text(
        5e-2,
        5e-5,
        r"$\mathcal{O}(\Delta t)$",
        fontsize=smaller_fontsize,
        ha='right',
        va='bottom'
    )

    # Grid and axis formatting
    ax.grid(which='major', linestyle='--', linewidth=1)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=smaller_fontsize)
    ax.set_xlabel(r'$\Delta t$', fontsize=larger_fontsize)
    ax.set_ylabel(r'$\bar{{e}}_{{r}}$', fontsize=larger_fontsize)

    # # Legend
    # fig.legend(
    #     legend_elements,
    #     legend_labels,
    #     bbox_to_anchor=(1.9, 0.5),
    #     loc='center right',
    #     ncol=1,
    #     fontsize=smaller_fontsize
    # )

    plt.tight_layout()
    out_file = os.path.join(
        plots_dir, f'{problem_name}_convergence_combined.pdf'
    )
    plt.savefig(out_file, bbox_inches='tight')


if __name__ == "__main__":
    main()
