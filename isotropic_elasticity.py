import streamlit as st
import numpy as np
import pandas as pd
import base64

# Function to compute Lame's parameters
def compute_lames_parameters(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu

# Function to compute stiffness matrix
def compute_stiffness_matrix(lam, mu):
    C = np.zeros((6, 6))
    C[0, 0] = lam + 2 * mu
    C[1, 1] = lam + 2 * mu
    C[2, 2] = lam + 2 * mu
    C[0, 1] = lam
    C[1, 0] = lam
    C[0, 2] = lam
    C[2, 0] = lam
    C[1, 2] = lam
    C[2, 1] = lam
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C

# Material data
materials_data = {
    "AlN": {"E": 300E9, "nu": 0.24},
    "B0.3Er0.5Al0.2N": {"E": 225E9, "nu": 0.24}
}

# Streamlit UI
st.title("Stiffness Matrix Calculation for Isotropy Assumption")

# Description of the theory
st.write(
    """
    ## Theory     
      
    For isotropic materials, 
    the stiffness matrix $\\mathbf{C_{ijkl}}$ can be expressed using Lame's parameters $ \\lambda $ (lambda) and $ \\mu $ (mu), 
    which are derived from the material's Young's modulus $ E $ and Poisson's ratio $ \\nu $ (nu). It is necessary to highlight the 
    relationship of bulk modulus ($K$) and shear modulus ($G$) with $E$ and $\\nu$. Mathematically, $ K = \\frac{E}{3(1-2\\nu)} $, and 
    $ G = \\frac{E}{2(1+\\nu)} $.
    
    Lame's parameters are defined as follows:
    
    1. **Lambda ($\\lambda$)**: Also known as the first Lame parameter, it represents the material's resistance to volume change under stress. 
    It is related to the bulk modulus $ K $ . Mathematically, $ \\lambda = \\frac{E \\cdot \\nu}{(1 + \\nu)(1 - 2\\nu)} $, where $ E $ is the Young's modulus and $ \\nu $ is Poisson's ratio.
    
    2. **Mu ($\\mu$)**: Also known as the shear modulus or second Lame parameter, it represents the material's resistance to shear deformation. 
    It is related to the shear modulus $ G $. Mathematically, $ \\mu = \\frac{E}{2(1 + \\nu)} $.
    
    The elements (annotated in matrix notation) of  stiffness matrix  $ \\mathbf{C} $  is then computed using Lame's parameters as follows:
    
    $$
    \\begin{align*}
    C_{11} & = \\lambda + 2\\mu; \\
    C_{22} & = \\lambda + 2\\mu; \\
    C_{33} & = \\lambda + 2\\mu; \\
    C_{12} & = C_{21} = \\lambda; \\
    C_{13} & = C_{31} = \\lambda; \\
    C_{23} & = C_{32} = \\lambda; \\
    C_{44} & = C_{55} = C_{66} = \\mu
    \\end{align*}
    $$
    
    where $ C_{ij} $ represents the stiffness matrix elements.
    
    Once the stiffness matrix is computed, it can be displayed in GPa for easier interpretation 
    and downloaded as a CSV file for further analysis or use in FEM based simulations.
    
    ### Mapping between Matrix Notation and $ijkl$ Notation
    
    In the stiffness matrix $ \\mathbf{C} $, the elements are arranged in matrix notation. However, they correspond to specific indices in the $ijkl$ notation of the stiffness tensor. 
    The mapping between matrix notation and $ijkl$ notation is as follows:
    
    - $ C_{11} $ corresponds to $ C_{1111} $
    - $ C_{22} $ corresponds to $ C_{2222} $
    - $ C_{33} $ corresponds to $ C_{3333} $
    - $ C_{12} = C_{21} $ corresponds to $ C_{1212} = C_{2121} $
    - $ C_{13} = C_{31} $ corresponds to $ C_{1313} = C_{3131} $
    - $ C_{23} = C_{32} $ corresponds to $ C_{2323} = C_{3232} $
    - $ C_{44} = C_{55} = C_{66}  $ corresponds to $ C_{4444} = C_{5555} = C_{6666} $
    """
)

# Select material
selected_material = st.selectbox("Select Material", list(materials_data.keys()))

# Compute Lame's parameters
E = materials_data[selected_material]["E"]
nu = materials_data[selected_material]["nu"]
lam, mu = compute_lames_parameters(E, nu)

# Compute stiffness matrix
C = compute_stiffness_matrix(lam, mu)

# Display Lame's parameters
st.subheader(f"Lame's Parameters of {selected_material}:")
st.write(f"Lambda (λ): {lam:.2e} Pa")
st.write(f"Mu (μ): {mu:.2e} Pa")

# Display stiffness matrix
st.subheader(f"Stiffness Matrix of {selected_material} (in GPa):")
stiffness_matrix_gpa = C / 1e9  # Convert to GPa
st.write(stiffness_matrix_gpa)

# Create DataFrame for CSV download
df = pd.DataFrame(stiffness_matrix_gpa)

# Allow download of properties in CSV format
st.markdown(f"### Download Isotropic Stiffness Tensor of {selected_material}")
csv = df.to_csv(index=False, header=False)  # Exclude header
b64 = base64.b64encode(csv.encode()).decode()
file_name = f"{selected_material}_cijkl_isotropy.csv"
href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV File</a>'
st.markdown(href, unsafe_allow_html=True)

