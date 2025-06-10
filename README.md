# FD-PyIMEX-RB: Implicit-Explicit Reduced Basis for Time-Dependent Problems

This project implements the IMEX-RB (Implicit-Explicit Reduced Basis) method to efficiently solve time-dependent problems discretized in space using finite difference (FD) schemes. This work accompanies the publication (...).

## Repository Structure

```
FD-PyIMEX-RB/
├── src/
│   ├── imexrb.py         # Implementation of the IMEX-RB algorithm.
│   ├── euler.py          # Classic backward and forward Euler methods.
│   └── problems1D.py      
├── utils/
│   ├── errors.py
│   └── mpl.pubstyle.py          
├── tutorials/
│   └── heat1D.ipynb      # Example notebook
├── requirements.txt       # Python dependencies
└── README.md           
```

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the required dependencies.
We recommend creating a virtual environment with:

```bash
python -m venv venv-imexrb
source venv-imexrb/bin/activate  
```

Install the necessary packages with:

```bash
pip install -r requirements.txt
```

### Running the Simulations -- Reproducibility

To reproduce the results from our analysis, navigate the directory numerical_tests, enter e.g. AdvDiff2D,
and run the scripts test_*.py. These will create a dir __RESULTS/ inside the repo. Once the tests are finished,
navigate the dir, e.g, postprocessing/AdvDiff2D. Here, run the jupyter notebook correspoding to the test you run.
Inside postprocessing/AdvDiff2D/plots you will obtain the plots you can find in the article.


execute the following notebooks:

- **tutorials/heat1D.ipynb** — a specific case study example on heat conduction


## Code Organization and Usage

- **src/imexrb.py**: Contains the implementation of the IMEX-RB method.
- **src/euler.py**: Implements the backward and forward Euler time integration schemes.
- **utils/problems1D.py**: Defines 1D problem classes for simulation purposes.
- **utils/utils.py**: Provides various utility functions to aid in numerical computations.


## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{imexrb,
  title={A self--adaptive Implicit--Explicit time integration method by reduced bases},
  author={Name and Collaborators},
  journal={Journal Name},
  year={2025},
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This work was supported by the Swiss National Science Foundation (grant agreement No 200021 197021, “Data-driven approximation of hemodynamics by combined reduced order modeling and deep neural networks”) and conducted at EPFL. We gratefully acknowledge the contributions of all collaborators.
