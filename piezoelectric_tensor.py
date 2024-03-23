import streamlit as st
import pandas as pd

# Define piezoelectric tensor coefficients for each material
materials_data = {
    "undoped AlN": [
        [0, 0, 0, 0, -0.2893, 0],
        [0, 0, 0, -0.2893, 0, 0],
        [-0.5801, -0.5801, 1.461, 0, 0, 0]
    ],
    "B0.3Er0.5Al0.2N alloy, unrotated": [
        [0, 0, 0, 0, 0.2283, 0],
        [0, 0, 0, 0.2283, 0, 0],
        [0.0943, 0.0943, 4.363, 0, 0, 0]
    ],
    "B0.3Er0.5Al0.2N alloy, rotated at orientation 1": [
        [0, 0.08, 0, -0.0808, 0.459, -0.4554],
        [-0.1882, -4.5632, -4.3232, 8.7947, -0.0807, 0.146],
        [0.1899, 4.3775, 4.3636, -8.6464, 0, -0.0667]
    ],
    "B0.3Er0.5Al0.2N alloy, rotated at orientation 2": [
        [1.1354, 0.947, 0.9856, -1.825, -2.005, 1.9339],
        [0.9564, 1.098, 0.9799, -1.9783, -1.8336, 1.93],
        [-0.9946, -0.9778, -1.1056, 1.9799, 2.0, -1.838]
    ]
}

# Display the piezoelectric tensor data for each material
st.title("Piezoelectric Tensors for B and Er doped, and Crystallographically Transformed AlN Materials")

# Display dropdown to select material
selected_material = st.selectbox("Select Material", list(materials_data.keys()))

# Display the selected piezoelectric tensor data
st.header(selected_material)
data = materials_data[selected_material]
html_table = "<table>"
for row in data:
    html_table += "<tr>"
    for value in row:
        html_table += f"<td>{value}</td>"
    html_table += "</tr>"
html_table += "</table>"
st.write(html_table, unsafe_allow_html=True)

# Download button
df = pd.DataFrame(data)
csv_data = df.to_csv(index=False, header=False)
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{selected_material.replace(' ', '_')}_piezoelectric_tensor.csv",
    mime="text/csv"
)

# Show array shape and number of elements
st.write(f"Shape of the array: {len(data)}x{len(data[0])}")
st.write(f"Number of elements: {len(data) * len(data[0])}")

# Access individual elements of the array
element_idx = st.text_input("Enter element index (e.g., a11):")
if element_idx and element_idx.startswith("a"):
    indices = tuple(map(int, element_idx[1:]))
    if 0 < indices[0] <= len(data) and 0 < indices[1] <= len(data[0]):
        value = data[indices[0] - 1][indices[1] - 1]
        st.write(f"Value at {element_idx}: {value} C/m$^{2}$")
    else:
        st.write("Invalid element index. Please enter valid index.")

