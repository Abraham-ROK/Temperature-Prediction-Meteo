import streamlit as st
import pydeck as pdk 

@st.cache(allow_output_mutation=True)
def display_map(clean_data):

    return pdk.Deck(
    #map_style='mapbox://styles/mapbox/light-v9',
    tooltip = {'html':'<b> Nomber of Observation:<b> {elevationValue}'},
    initial_view_state=pdk.ViewState(
        latitude=46.2276,
        longitude=2.2137,
        zoom=4.65, 
        #min_zoom=5, 
        #max_zoom=15, 
        pitch=50,
        bearing=-27.40,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data = clean_data[['Longitude','Latitude']],
                #data = ndf,
                get_position='[Longitude, Latitude]',
                auto_highlight=True,
                #radius=200,

                elevation_scale=50,
                pickable=True,

                elevation_range=[0, 3000],
                extruded=True,
                coverage=1,
                ),
                
                    ],
)