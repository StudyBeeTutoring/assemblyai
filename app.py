# app.py
import streamlit as st
import joblib
import plotly.express as px
import pandas as pd  # We need pandas for one small data manipulation

# --- CONFIGURATION ---
# Set the page configuration as the very first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="The Assembly AI",
    page_icon="🌐"
)


@st.cache_data
def load_analysis():
    """Loads the pre-computed analysis results from a GitHub Release URL."""
    
    # --- IMPORTANT: PASTE THE URL YOU COPIED FROM GITHUB RELEASES HERE ---
    ANALYSIS_URL = "https://github.com/StudyBeeTutoring/assemblyai/releases/download/v2.0.0/un_analysis_results.joblib"

    try:
        # To load a joblib file from a URL, we first download its content
        response = requests.get(ANALYSIS_URL)
        response.raise_for_status() # This will raise an error if the download fails

        # We then write the downloaded content to a temporary file in memory
        with open("un_analysis_results.joblib", "wb") as f:
            f.write(response.content)
        
        # Now we can load the result from that temporary file
        results = joblib.load('un_analysis_results.joblib')
        return results

    except Exception as e:
        st.error(f"Error loading analysis file from URL: {e}")
        st.error(f"Please check the URL in your app.py: {ANALYSIS_URL}")
        return None

results_by_decade = load_analysis()
# --- UI & VISUALIZATION ---

st.title("🌐 The Assembly AI")
st.markdown(
    "An interactive analysis of voting blocs and political proximity in the UN General Assembly, from the 1940s to the 2020s. This tool uses unsupervised machine learning (K-Means Clustering & PCA) to discover hidden alliances based purely on voting records.")

# Check if the analysis file was loaded successfully
if results_by_decade is None:
    st.error(
        "Error: Analysis file ('un_analysis_results.joblib') not found. Please run the `analyze_blocs.py` script from Phase 2 first.")
else:
    # --- INTERACTIVE WIDGETS ---
    st.sidebar.header("Controls")
    # Get a sorted list of the decades we have data for
    decade_list = sorted(results_by_decade.keys())

    # Create a slider in the sidebar for the user to select a decade
    selected_decade = st.sidebar.select_slider(
        "Select a Decade to Analyze",
        options=decade_list,
        value=1980  # Default value when the app loads
    )

    st.header(f"Geopolitical Landscape of the {selected_decade}s")

    # --- DATA PREPARATION FOR VISUALIZATION ---
    # Get the specific DataFrame for the user-selected decade
    decade_results_df = results_by_decade[selected_decade]

    # Convert the numerical 'bloc' number to a more readable string for chart legends
    # e.g., cluster 0 becomes "Bloc 0"
    decade_results_df['bloc'] = "Bloc " + decade_results_df['bloc'].astype(str)

    # --- VISUALIZATION 1: INTERACTIVE WORLD MAP ---
    st.subheader("Discovered Voting Blocs on the World Map")
    st.markdown(
        "This map shows the automatically discovered geopolitical blocs. Countries with the same color have consistently similar voting patterns during this decade.")

    # Create a choropleth map using Plotly Express
    fig_map = px.choropleth(
        decade_results_df,
        locations="country_iso",  # Use the ISO alpha-3 codes for country locations
        color="bloc",  # Color each country based on its assigned bloc
        hover_name="country_iso",  # Show the country code when hovering
        color_discrete_sequence=px.colors.qualitative.Vivid,  # Use a vibrant color scheme
        title=f"Geopolitical Blocs in the {selected_decade}s"
    )

    # Customize the map's appearance
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'),
        legend_title_text='Voting Bloc'
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # --- VISUALIZATION 2: POLITICAL PROXIMITY SCATTER PLOT ---
    st.subheader(f"Political Proximity Map for the {selected_decade}s")
    st.markdown(
        "This chart visualizes the 'political distance' between countries, calculated using Principal Component Analysis (PCA). **Nations that are closer together on this chart have more similar voting records.** You can often see clear clusters representing major geopolitical alliances.")

    # Create an interactive scatter plot
    fig_scatter = px.scatter(
        decade_results_df,
        x='x',  # The x-coordinate from PCA
        y='y',  # The y-coordinate from PCA
        color='bloc',  # Color points by the same bloc as the map
        hover_name='country_iso',  # Show country code on hover
        title='2D Projection of Voting Similarity'
    )

    # Make the chart cleaner
    fig_scatter.update_traces(marker=dict(size=10))
    fig_scatter.update_layout(xaxis_title="Political Dimension 1", yaxis_title="Political Dimension 2")

    st.plotly_chart(fig_scatter, use_container_width=True)
