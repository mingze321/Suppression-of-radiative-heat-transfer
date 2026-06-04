# How to Use This Code

This script runs an inverse-design optimization for a multilayer Si/SiO₂ thin-film stack using a TensorFlow transfer-matrix model. It optimizes layer thicknesses, calculates the resulting absorption spectrum, plots the result, and saves the designed spectrum to a CSV file.

## 1. Prepare the folder

Put the Python script in a working folder with this structure:

```text
project_folder/
├── your_script.py
└── dielectric_functions/
    ├── Silicon_thin_film_NK_franta.txt
    └── SiO2_NK_franta.txt
```

The two material files must exist because the code reads them directly.

## 2. Install required packages

Use Python with TensorFlow installed. The script needs:

```bash
pip install tensorflow numpy scipy matplotlib
```

The code currently forces TensorFlow to use the CPU, so a GPU is not required.

## 3. Run the code

From the project folder, run:

```bash
python your_script.py
```

The script will print the TensorFlow version, train the model for multiple random trials, and report the best layer thicknesses found.

## 4. Main settings to change

Common parameters are near the bottom or inside `MyModel`:

```python
EPOCHS = 1500
number_of_trails = 10
```

## 5. Important notes

- The code uses Si and SiO₂ material data from the `dielectric_functions` folder.
- The active stack is defined in `dielectric1` inside `MyModel.call()`.
- The script uses the S-polarization transfer-matrix formulas by default. P-polarization formulas are present but commented out.

## 6. Citation and contact

- You can cite all versions by using the DOI 10.5281/zenodo.20549145. This DOI represents all versions, and will always resolve to the latest one. Read more.
- If you have any questions, or need help to use the code, please feel free to contact Dr. Mingze He,  hemingze1995@gmail.com
