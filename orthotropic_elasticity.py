import streamlit as st
import numpy as np
import pandas as pd
import base64

#def compute_compliance_matrix(E1, E2, E3, nu12, nu13, nu21, nu23, nu31, nu32):
    ## Compute the shear moduli ratios
    #G23 = E2 / (2 * (1 + nu23))
    #G31 = E3 / (2 * (1 + nu31))
    #G12 = E1 / (2 * (1 + nu12))
    
    ## Compute the compliance matrix
    #S = np.zeros((6, 6))
    #S[0, 0] = 1 / E1
    #S[0, 1] = -nu12 / E1
    #S[0, 2] = -nu13 / E1
    #S[1, 0] = -nu21 / E2
    #S[1, 1] = 1 / E2
    #S[1, 2] = -nu23 / E2
    #S[2, 0] = -nu31 / E3
    #S[2, 1] = -nu32 / E3
    #S[2, 2] = 1 / E3
    #S[3, 3] = 1 / G23
    #S[4, 4] = 1 / G31
    #S[5, 5] = 1 / G12
    
    #return S

def compute_compliance_matrix(E1, E2, E3, nu12, nu13, nu21, nu23, nu31, nu32):
    # Compute the shear moduli ratios
    G23 = E2 / (2 * (1 + nu23))
    G31 = E3 / (2 * (1 + nu31))
    G12 = E1 / (2 * (1 + nu12))
    
    # Compute the compliance matrix
    S11 = 1 / E1
    S12 = -nu12 / E1
    S13 = -nu13 / E1
    S21 = -nu21 / E2
    S22 = 1 / E2
    S23 = -nu23 / E2
    S31 = -nu31 / E3
    S32 = -nu32 / E3
    S33 = 1 / E3
    S44 = 1 / G23
    S55 = 1 / G31
    S66 = 1 / G12
    
    S = np.array([[S11, S12, S13, 0, 0, 0],
                  [S21, S22, S23, 0, 0, 0],
                  [S31, S32, S33, 0, 0, 0],
                  [0, 0, 0, S44, 0, 0],
                  [0, 0, 0, 0, S55, 0],
                  [0, 0, 0, 0, 0, S66]])
    
    return S

def compute_stiffness_matrix(S):
    C = np.linalg.inv(S)
    return C / 1e9

def download_csv(data, material_name):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False, header=False)
    b64 = base64.b64encode(csv.encode()).decode()
    file_name = f"{material_name}_stiffness_matrix.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV File</a>'
    return href

# Streamlit UI
st.title("Stiffness Matrix Calculation for Orthotropy Assumption")

# Material data
material_data = {
    "AlN": {"E1": 275E9, "E2": 300E9, "E3": 325E9, "nu12": 0.26, "nu13": 0.22, "nu21": 0.26, "nu23": 0.24, "nu31": 0.22, "nu32": 0.24},
    "B0.3Er0.5Al0.2N": {"E1": 250E9, "E2": 225E9, "E3": 275E9, "nu12": 0.22, "nu13": 0.26, "nu21": 0.22, "nu23": 0.24, "nu31": 0.26, "nu32": 0.24}
}

# Select material
selected_material = st.selectbox("Select Material", list(material_data.keys()))

# Compute compliance matrix
st.subheader(f"Compliance Matrix of {selected_material}  (in Pa$^{-1}$):")
compliance_matrix = compute_compliance_matrix(**material_data[selected_material])
st.write(pd.DataFrame(compliance_matrix).applymap(lambda x: f"{x:.2e}"))

# Compute stiffness matrix
stiffness_matrix = compute_stiffness_matrix(compliance_matrix)

# Display stiffness matrix
st.subheader(f"Stiffness Matrix of {selected_material} (in GPa):")
stiffness_matrix_gpa = stiffness_matrix.astype(float)  # Convert to float
st.write(stiffness_matrix_gpa)

# Download CSV file
download_link = download_csv(stiffness_matrix, selected_material)
st.markdown(download_link, unsafe_allow_html=True)

