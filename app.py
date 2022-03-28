import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/rf_site_deploy.joblib')

st.set_page_config(page_title="Site Energy Usage Intensity Prediction App",
                   page_icon="🚧", layout="wide")


options_facility = ['Grocery_store_or_food_market',
       'Warehouse_Distribution_or_Shipping_center',
       'Retail_Enclosed_mall', 'Education_Other_classroom',
       'Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage',
       'Office_Uncategorized', 'Data_Center', 'Commercial_Other',
       'Mixed_Use_Predominantly_Commercial',
       'Office_Medical_non_diagnostic', 'Education_College_or_university',
       'Industrial', 'Laboratory',
       'Public_Assembly_Entertainment_culture',
       'Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized',
       'Lodging_Hotel', 'Retail_Strip_shopping_mall',
       'Education_Uncategorized', 'Health_Care_Inpatient',
       'Public_Assembly_Drama_theater', 'Public_Assembly_Social_meeting',
       'Religious_worship', 'Mixed_Use_Commercial_and_Residential',
       'Office_Bank_or_other_financial', 'Parking_Garage',
       'Commercial_Unknown', 'Service_Vehicle_service_repair_shop',
       'Service_Drycleaning_or_Laundry', 'Public_Assembly_Recreation',
       'Service_Uncategorized', 'Warehouse_Refrigerated',
       'Food_Service_Uncategorized', 'Health_Care_Uncategorized',
       'Food_Service_Other', 'Public_Assembly_Movie_Theater',
       'Food_Service_Restaurant_or_cafeteria', 'Food_Sales',
       'Public_Assembly_Uncategorized', 'Nursing_Home',
       'Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare',
       '5plus_Unit_Building', 'Multifamily_Uncategorized',
       'Lodging_Dormitory_or_fraternity_sorority',
       'Public_Assembly_Library', 'Public_Safety_Uncategorized',
       'Public_Safety_Fire_or_police_station', 'Office_Mixed_use',
       'Public_Assembly_Other', 'Public_Safety_Penitentiary',
       'Health_Care_Outpatient_Uncategorized', 'Lodging_Other',
       'Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse',
       'Public_Assembly_Stadium', 'Lodging_Uncategorized',
       '2to4_Unit_Building', 'Warehouse_Uncategorized']

options_state = ['State_1', 'State_2', 'State_4', 'State_6', 'State_8', 'State_10', 'State_11']

options_building_class = ['Commercial', 'Residential']

st.markdown("<h1 style='text-align: center;'>ASite Energy Usage Intensity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        energy_star_rating = st.slider("Energy Star Rating: ", 1, 100, value=1, format="%d")
        facility_type = st.selectbox("Select Facility Type: ", options=options_facility)
        floor_area = st.text_input("Enter Floor Area", value="", max_chars=6)
        year_built = st.slider("Year built: ", 1600, 2015, value=1600, format="%d")
        State_Factor = st.selectbox("Select State Code: ", options=options_state)
        building_class = st.selectbox("Select Building Type: ", options=options_building_class)
        ELEVATION = st.text_input("Elevation", value="", max_chars=6)
        submit = st.form_submit_button("Predict")


    if submit:
        facility_type = ordinal_encoder(facility_type, options_facility)
        State_Factor = ordinal_encoder(State_Factor, options_state)
        building_class = ordinal_encoder(building_class, options_building_class)

        data = np.array([energy_star_rating,facility_type,floor_area,year_built,State_Factor,building_class,ELEVATION]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted Energy Usage Intensity is:  {pred}")

if __name__ == '__main__':
    main()
