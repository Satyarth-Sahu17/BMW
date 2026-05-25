import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from PIL import Image
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_page_config(
    page_title="BMW Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.set_page_config(
    page_title="BMW Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"> BioAcoustic Wildlife Monitoring Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", "http://localhost:8000")
    
    st.markdown("---")
    st.header("Quick Stats")
    
    # Mock stats
    st.metric("Total Recordings", "1,247")
    st.metric("Species Detected", "5")
    st.metric("Threats Detected", "12")
    
    st.markdown("---")
    st.header("Filters")
    date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=7), datetime.now()])
    species_filter = st.multiselect("Species", ["wolf", "tiger", "gunshot", "chainsaw", "background"])
    confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.7)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["Live Analysis", "Historical Data", "Geographic View", "Threat Alerts"])

with tab1:
    st.header("Analyze Audio Recording")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac'],
            help="Upload an audio recording for analysis"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("Analyze Recording", type="primary"):
                with st.spinner("Analyzing audio..."):
                    try:
                        # Call API
                        files = {"file": uploaded_file.getvalue()}
                        response = requests.post(
                            f"{api_url}/predict",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Analysis Complete!")
                            
                            # Display results
                            pred = result['predictions'][0]
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Detected", pred['label'].upper())
                            with col_b:
                                st.metric("Confidence", f"{pred['score']:.2%}")
                            with col_c:
                                st.metric("Duration", f"{pred['end'] - pred['start']:.1f}s")
                            
                            # Show all class probabilities
                            if 'class_probabilities' in result:
                                st.subheader("Class Probabilities")
                                probs_df = pd.DataFrame([
                                    {"Class": k, "Probability": v}
                                    for k, v in result['class_probabilities'].items()
                                ]).sort_values('Probability', ascending=False)
                                
                                fig = px.bar(probs_df, x='Class', y='Probability',
                                           color='Probability',
                                           color_continuous_scale='Blues')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show metadata
                            with st.expander("Metadata"):
                                st.json(result.get('metadata', {}))
                        else:
                            st.error(f"API Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Make sure it's running on the specified URL.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. Upload an audio file
        2. Click 'Analyze Recording'
        3. View detection results
        
        **Supported Formats:**
        - WAV
        - MP3
        - FLAC
        
        **Detected Classes:**
        - 🐺 Wolf
        - 🐅 Tiger
        - 🔫 Gunshot
        - 🪚 Chainsaw
        - 🌲 Background noise
        """)

with tab2:
    st.header("Historical Detections")
    
    # Generate mock historical data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    mock_data = pd.DataFrame({
        'date': dates,
        'wolf': np.random.randint(0, 10, len(dates)),
        'tiger': np.random.randint(0, 5, len(dates)),
        'gunshot': np.random.randint(0, 3, len(dates)),
        'chainsaw': np.random.randint(0, 2, len(dates)),
        'background': np.random.randint(50, 100, len(dates))
    })
    
    # Activity timeline
    st.subheader("Detection Timeline")
    fig = go.Figure()
    for col in ['wolf', 'tiger', 'gunshot', 'chainsaw']:
        fig.add_trace(go.Scatter(
            x=mock_data['date'],
            y=mock_data[col],
            mode='lines+markers',
            name=col.capitalize()
        ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Detections",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Activity Heatmap")
    heatmap_data = mock_data.set_index('date')[['wolf', 'tiger', 'gunshot', 'chainsaw']]
    fig = px.imshow(
        heatmap_data.T,
        labels=dict(x="Date", y="Species", color="Detections"),
        aspect="auto",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent detections table
    st.subheader("Recent Detections")
    recent_detections = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=20, freq='6H'),
        'Species': np.random.choice(['Wolf', 'Tiger', 'Gunshot', 'Chainsaw'], 20),
        'Confidence': np.random.uniform(0.7, 0.99, 20),
        'Recorder': [f'REC-{np.random.randint(1, 5):02d}' for _ in range(20)]
    }).sort_values('Timestamp', ascending=False)
    
    st.dataframe(recent_detections, use_container_width=True)
    
    # Download button
    csv = recent_detections.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        "detections.csv",
        "text/csv",
        key='download-csv'
    )

with tab3:
    st.header("Geographic Distribution")
    
    # Generate mock geographic data
    n_points = 50
    mock_map_data = pd.DataFrame({
        'lat': np.random.uniform(27.9, 28.1, n_points),
        'lon': np.random.uniform(86.8, 87.0, n_points),
        'species': np.random.choice(['Wolf', 'Tiger', 'Gunshot', 'Chainsaw'], n_points),
        'confidence': np.random.uniform(0.7, 0.99, n_points),
        'timestamp': pd.date_range(end=datetime.now(), periods=n_points, freq='12H')
    })
    
    # Map visualization
    st.subheader("Detection Locations")
    
    color_map = {
        'Wolf': '#3498db',
        'Tiger': '#e74c3c',
        'Gunshot': '#e67e22',
        'Chainsaw': '#95a5a6'
    }
    mock_map_data['color'] = mock_map_data['species'].map(color_map)
    
    st.map(mock_map_data[['lat', 'lon']])
    
    # Species distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Species Distribution")
        species_counts = mock_map_data['species'].value_counts()
        fig = px.pie(
            values=species_counts.values,
            names=species_counts.index,
            title="Detection Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confidence Distribution")
        fig = px.histogram(
            mock_map_data,
            x='confidence',
            nbins=20,
            title="Confidence Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Threat Detection Alerts")
    
    st.warning("Threat detection monitoring is active")
    
    # Mock threat alerts
    threats = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=12, freq='2D'),
        'Type': np.random.choice(['Gunshot', 'Chainsaw'], 12),
        'Confidence': np.random.uniform(0.85, 0.99, 12),
        'Location': [f'Zone {chr(65 + i % 5)}' for i in range(12)],
        'Status': np.random.choice(['Verified', 'Under Review', 'False Alarm'], 12),
    }).sort_values('Timestamp', ascending=False)
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Threats (30d)", len(threats), delta=3)
    with col2:
        verified = (threats['Status'] == 'Verified').sum()
        st.metric("Verified Threats", verified, delta=-1, delta_color="inverse")
    with col3:
        under_review = (threats['Status'] == 'Under Review').sum()
        st.metric("Under Review", under_review)
    
    # Threat timeline
    st.subheader("Threat Timeline")
    threat_timeline = threats.groupby([pd.Grouper(key='Timestamp', freq='D'), 'Type']).size().reset_index(name='count')
    fig = px.bar(threat_timeline, x='Timestamp', y='count', color='Type',
                title="Daily Threat Detections",
                color_discrete_map={'Gunshot': '#e74c3c', 'Chainsaw': '#e67e22'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts table
    st.subheader("Recent Alerts")
    
    def status_color(status):
        if status == 'Verified':
            return '🔴'
        elif status == 'Under Review':
            return '🟡'
        else:
            return '🟢'
    
    threats['Status Icon'] = threats['Status'].apply(status_color)
    st.dataframe(
        threats[['Status Icon', 'Timestamp', 'Type', 'Confidence', 'Location', 'Status']],
        use_container_width=True
    )
    
    # Alert notification settings
    with st.expander("Alert Settings"):
        st.subheader("Notification Preferences")
        st.checkbox("Email notifications", value=True)
        st.checkbox("SMS alerts", value=False)
        st.slider("Alert threshold confidence", 0.5, 1.0, 0.85)
        st.multiselect("Alert types", ["Gunshot", "Chainsaw"], default=["Gunshot", "Chainsaw"])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    BMW Dashboard v1.0 | BioAcoustic Monitoring of Endangered Wildlife<br>
    For support, contact: support@bmw-monitoring.org
</div>
""", unsafe_allow_html=True)
