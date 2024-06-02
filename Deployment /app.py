import gradio as gr
import pandas as pd
import joblib

NA = ['BrDale', 'BrkSide', 'Edwards', 'IDOTRR', 'MeadowV', 'NAmes', 'OldTown', 'SWISU', 'Sawyer']
NB = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'SawyerW', 'Somerst', 'Timber', 'Veenker']
NC = ['NoRidge', 'NridgHt', 'StoneBr']
ND = ['Mitchel', 'NPkVill', 'Blueste']

def calc(OverallQual, MasVnrArea, _1stFlrSF, GrLivArea, GarageYrBlt, GarageCars, GarageArea, Neighborhood, ExterQual, KitchenQual):
    # Load the trained model
    loaded_rf = joblib.load(open('random_forest_model.pkl', 'rb'))

    # Define the features used during training
    features = ['OverallQual', 'MasVnrArea', '1stFlrSF', 'GrLivArea', 'GarageYrBlt',
                'GarageCars', 'GarageArea', 'Neighborhood_A',
                'Neighborhood_B', 'Neighborhood_C', 'Neighborhood_D', 'ExterQual_Ex',
                'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'KitchenQual_Ex',
                'KitchenQual_Gd', 'KitchenQual_TA_Fa']

    Neighborhood_A = 1 if Neighborhood in NA else 0
    Neighborhood_B = 1 if Neighborhood in NB else 0
    Neighborhood_C = 1 if Neighborhood in NC else 0
    Neighborhood_D = 1 if Neighborhood in ND else 0

    ExterQual_Ex = 1 if ExterQual == 'Ex' else 0
    ExterQual_Fa = 1 if ExterQual == 'Fa' else 0
    ExterQual_Gd = 1 if ExterQual == 'Gd' else 0
    ExterQual_TA = 1 if ExterQual == 'TA' else 0
    KitchenQual_Ex = 1 if KitchenQual == 'Ex' else 0
    KitchenQual_Gd = 1 if KitchenQual == 'Gd' else 0
    KitchenQual_TA_Fa = 1 if KitchenQual == 'TA' or KitchenQual == 'Fa' else 0

    # Create a new DataFrame with the example values
    new_data = pd.DataFrame({
        'OverallQual': [OverallQual],
        'MasVnrArea': [MasVnrArea],
        '1stFlrSF': [_1stFlrSF],
        'GrLivArea': [GrLivArea],
        'GarageYrBlt': [GarageYrBlt],
        'GarageCars': [GarageCars],
        'GarageArea': [GarageArea],
        'Neighborhood_A': [Neighborhood_A],
        'Neighborhood_B': [Neighborhood_B],
        'Neighborhood_C': [Neighborhood_C],
        'Neighborhood_D': [Neighborhood_D],
        'ExterQual_Ex': [ExterQual_Ex],
        'ExterQual_Fa': [ExterQual_Fa],
        'ExterQual_Gd': [ExterQual_Gd],
        'ExterQual_TA': [ExterQual_TA],
        'KitchenQual_Ex': [KitchenQual_Ex],
        'KitchenQual_Gd': [KitchenQual_Gd],
        'KitchenQual_TA_Fa': [KitchenQual_TA_Fa]
    })

    # Make predictions on the new data
    prediction = loaded_rf.predict(new_data)

    # Return or use the predictions as needed
    return f"Predicted Price: ${prediction[0]:,.2f}"


demo = gr.Interface(
    fn=calc,
    inputs=[
        gr.Textbox(label="Overall Quality (Eg: 7-56)", type="text", elem_classes="inputs"),
        gr.Textbox(label="Masonry veneer area in square feet (Eg: 0-1600)", type="text", elem_classes="inputs"),
        gr.Textbox(label="First Floor square feet (Eg: 334-4692)", type="text", elem_classes="inputs"),
        gr.Textbox(label="Above grade (ground) living area square feet", type="text", elem_classes="inputs"),
        gr.Textbox(label="Year garage was built (Eg: 1900-2010)", type="text", elem_classes="inputs"),
        gr.Textbox(label="Size of garage in car capacity (Eg: 0-4)", type="text", elem_classes="inputs"),
        gr.Textbox(label="Size of garage in square feet (Eg: 0-1418)", type="text", elem_classes="inputs"),
        gr.Dropdown(label="Physical locations within Ames city limits", choices=['BrDale', 'BrkSide', 'Edwards', 'IDOTRR', 'MeadowV', 'NAmes', 'OldTown', 'SWISU', 'Sawyer', 'Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'SawyerW', 'Somerst', 'Timber', 'Veenker', 'NoRidge', 'NridgHt', 'StoneBr', 'Mitchel', 'NPkVill', 'Blueste'], elem_classes="inputs"),
        gr.Dropdown(label="Exterior material quality", choices=['Gd', 'TA', 'Ex', 'Fa'], elem_classes="inputs"),
        gr.Dropdown(label="Kitchen quality", choices=['Gd', 'TA', 'Ex', 'Fa'], elem_classes="inputs"),
    ],
    outputs=gr.Textbox(label="Predicted Price", type="text", elem_classes="outputs"),  # Labeling the output as "Predicted Price"
    description="""
    ## Welcome to house price prediction
    **Please enter your data in the following format:**
    - **OverallQual**: Overall Quality (Eg:7-56)
    - **MasVnrArea**: Masonry veneer area in square feet (Eg:0-1600)
    - **1stFlrSF**: First Floor square feet (Eg:334-4692)
    - **GrLivArea**: Above grade (ground) living area square feet
    - **GarageYrBlt**: Year garage was built (Eg:1900-2010)
    - **GarageCars**: Size of garage in car capacity (Eg:0-4)
    - **GarageArea**: Size of garage in square feet (Eg:0-1418)
    - **Neighborhood**: Physical locations within Ames city limits
    - **ExterQual**: Exterior material quality (Options: 'Gd', 'TA', 'Ex', 'Fa')
    - **KitchenQual**: Kitchen quality (Options: 'Gd', 'TA', 'Ex', 'Fa')
    """,
    theme="soft",
    css="styles.css",
    # Add an image component
    examples=[
        ["8", "200", "1500", "2000", "2005", "2", "500", "NAmes", "Gd", "Ex"],
    ],
)
gr.HTML("<img src='https://i.ibb.co/6WmGjzL/modern-home-beautiful-3d.jpg' />")
# if __name__ == "__main__":
demo.launch()
