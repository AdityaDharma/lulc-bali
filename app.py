import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import json
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model
import tempfile
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config with custom styling
st.set_page_config(
    page_title="Land Cover Monitoring",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        --success-gradient: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        --accent-gradient: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        --dark-gradient: linear-gradient(135deg, #232526 0%, #414345 100%);
        --glass-bg: rgba(255, 255, 255, 0.15);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-primary: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.1);
        --border-radius: 20px;
        --border-radius-sm: 12px;
    }
    
    /* Main container styling */
    .main {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Background pattern */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* Custom header styling with glassmorphism effect */
    .header-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: var(--shadow-primary);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='9' cy='9' r='9'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        background: linear-gradient(45deg, #ffffff, #e8e8e8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Modern card styling with glassmorphism */
    .info-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-soft);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .info-card h3 {
        margin-top: 0;
        color: #2d3748;
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, 
                    rgba(102, 126, 234, 0.9) 0%, 
                    rgba(118, 75, 162, 0.9) 50%,
                    rgba(255, 94, 77, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: var(--border-radius-sm);
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-primary);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
        opacity: 0;
    }
    
    .metric-card:hover::before {
        animation: shimmer 0.8s ease-in-out;
        opacity: 1;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.95;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Upload area styling with modern gradient */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: var(--border-radius);
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, 
                    rgba(248, 249, 255, 0.8) 0%, 
                    rgba(232, 236, 255, 0.8) 100%);
        backdrop-filter: blur(10px);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, 
                    rgba(102, 126, 234, 0.1) 0%, 
                    rgba(118, 75, 162, 0.1) 100%);
        transform: translateY(-2px);
    }
    
    /* Enhanced success message */
    .success-message {
        background: linear-gradient(135deg, 
                    rgba(132, 250, 176, 0.9) 0%, 
                    rgba(143, 211, 244, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(132, 250, 176, 0.3);
        color: #1a365d;
        padding: 1.5rem;
        border-radius: var(--border-radius-sm);
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: var(--shadow-soft);
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, 
                    rgba(248, 249, 255, 0.95) 0%, 
                    rgba(232, 236, 255, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Class legend with modern styling */
    .class-legend {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: var(--border-radius-sm);
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .class-legend:hover {
        background: rgba(255, 255, 255, 0.8);
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .class-color {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        margin-right: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.8);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: all 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #7c8ef0 0%, #8457b8 100%);
    }
    
    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .stProgress > div > div > div {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: var(--border-radius-sm);
        border: 2px solid rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, 
                    rgba(248, 249, 255, 0.9) 0%, 
                    rgba(232, 236, 255, 0.9) 100%);
        backdrop-filter: blur(20px);
        box-shadow: var(--shadow-soft);
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: var(--border-radius-sm);
        overflow: hidden;
        box-shadow: var(--shadow-soft);
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
    }
    
    /* Enhanced download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8b4513;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.4);
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(252, 182, 159, 0.6);
        background: linear-gradient(135deg, #ffe0c4 0%, #faa687 100%);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c8ef0 0%, #8457b8 100%);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2.5rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .info-card {
            padding: 1.5rem;
        }
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, 
                    rgba(102, 126, 234, 0.1) 0%, 
                    rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        padding: 2rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with enhanced styling
st.markdown("""
<div class="header-container">
    <div class="header-title">🛰️ Land Cover Classification</div>
    <div class="header-subtitle">Advanced satellite imagery analysis</div>
</div>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_classification_model():
    """Load the trained model"""
    try:
        model_path = 'aditya_DL3.keras'
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_landcover_metadata():
    """Load land cover class metadata"""
    lc_data = {
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "palette": ["#006400", "#228B22", "#32CD32", "#D2B48C", "#ADFF2F", 
                   "#87CEFA", "#FFD700", "#FFA500", "#F08080"],
        "label": ["🌲 Dryland forest", "🌿 Mangrove forest", "🌱 Plantation forest", 
                 "🏜️ Bare land", "🌾 Savanna and grasses", "💧 Waterbody", 
                 "🚜 Dry agriculture", "🌾 Paddy field", "🏢 Built-up"]
    }
    
    lc_df = pd.DataFrame(lc_data)
    values = lc_df["values"].to_list()
    palette = lc_df["palette"].to_list()
    labels = lc_df["label"].to_list()
    
    cmap = ListedColormap(palette)
    patches = [mpatches.Patch(color=palette[i], label=labels[i]) for i in range(len(values))]
    legend = {
        "handles": patches,
        "bbox_to_anchor": (1.05, 1),
        "loc": 2,
        "borderaxespad": 0.0,
    }
    
    return lc_df, cmap, legend, labels

def create_plotly_visualization(rgb_image, pred_label, labels, palette):
    """Create interactive Plotly visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📡 Original Satellite Image", "🗺️ Land Cover Classification"),
        specs=[[{"type": "image"}, {"type": "image"}]]
    )
    
    # Original image
    fig.add_trace(
        go.Image(z=rgb_image),
        row=1, col=1
    )
    
    # Classification result with custom colorscale
    unique_classes = np.unique(pred_label)
    unique_classes = unique_classes[unique_classes != 0]
    
    if len(unique_classes) > 0:
        colorscale = []
        min_class = min(unique_classes)
        max_class = max(unique_classes)
        
        for i, cls in enumerate(unique_classes):
            normalized_pos = (cls - min_class) / (max_class - min_class) if max_class != min_class else 0
            if cls <= len(palette):
                colorscale.append([normalized_pos, palette[cls-1]])
        
        if len(colorscale) == 1:
            colorscale = [[0, colorscale[0][1]], [1, colorscale[0][1]]]
    else:
        colorscale = [[0, '#006400'], [1, '#F08080']]
    
    fig.add_trace(
        go.Heatmap(
            z=pred_label,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Land Cover Class",
                titleside="right",
                tickmode="array",
                tickvals=list(unique_classes),
                ticktext=[labels[cls-1] if cls <= len(labels) else f"Class {cls}" for cls in unique_classes],
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(102, 126, 234, 0.3)",
                borderwidth=1
            ),
            hovertemplate="<b>Class:</b> %{z}<br><b>Row:</b> %{y}<br><b>Col:</b> %{x}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="",
        font=dict(size=12, family="Inter"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

def create_area_chart(area_stats, labels, palette):
    """Create interactive area statistics chart"""
    if not area_stats:
        return None
    
    classes = list(area_stats.keys())
    areas = [area_stats[cls]['area_ha'] for cls in classes]
    class_labels = [area_stats[cls]['label'] for cls in classes]
    colors = [palette[cls-1] for cls in classes if cls <= len(palette)]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=class_labels,
            values=areas,
            marker_colors=colors,
            hole=0.5,
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=12, family="Inter"),
            hovertemplate="<b>%{label}</b><br>" +
                         "Area: %{value:.2f} ha<br>" +
                         "Percentage: %{percent}<br>" +
                         "<extra></extra>",
            marker=dict(
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="📊 Land Cover Distribution",
            font=dict(size=18, family="Inter", color="#2d3748"),
            x=0.5
        ),
        font=dict(size=12, family="Inter"),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_matplotlib_visualization(rgb_image, pred_label, cmap, legend):
    """Create matplotlib visualization as fallback"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # RGB Image
    ax1.imshow(rgb_image)
    ax1.set_title("📡 Original Satellite Image", fontsize=16, pad=20, fontweight='bold')
    ax1.axis('off')
    
    # Prediction
    im = ax2.imshow(pred_label, cmap=cmap, vmin=1, vmax=9)
    ax2.set_title("🗺️ Land Cover Classification", fontsize=16, pad=20, fontweight='bold')
    ax2.axis('off')
    
    # Add legend
    ax2.legend(**legend)
    
    plt.tight_layout()
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    
    return fig

def process_image(uploaded_file, model, cmap, legend, labels):
    """Process uploaded image and make predictions"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with rio.open(tmp_path) as src:
            test_patch = src.read()
        
        os.unlink(tmp_path)
        
        test_patch = np.transpose(test_patch, (1, 2, 0))
        
        if test_patch.shape[2] < 3:
            st.error("❌ Image must have at least 3 bands for RGB visualization")
            return None, None, None, None
        
        if test_patch.shape[2] >= 3:
            rgb_image = test_patch[:, :, [2, 1, 0]]
            rgb_image = np.clip(rgb_image, 0, np.percentile(rgb_image, 98))
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        else:
            rgb_image = test_patch[:, :, 0]
        
        if len(test_patch.shape) == 3:
            pred_patch = model.predict(test_patch[np.newaxis, ...])
        else:
            st.error("❌ Invalid image dimensions")
            return None, None, None, None
        
        pred_label = np.argmax(pred_patch[0], axis=-1)
        
        pixel_size = st.session_state.get('pixel_size', 30)
        pixel_area = pixel_size * pixel_size
        
        classes = np.unique(pred_label)
        classes = classes[classes != 0]
        
        area_stats = {}
        for cls in classes:
            cls_int = int(cls)
            
            if cls_int <= len(labels):
                pixel_count = int(np.sum(pred_label == cls_int))
                area_m2 = pixel_count * pixel_area
                area_ha = area_m2 / 10000
                area_stats[cls_int] = {
                    "label": labels[cls_int - 1] if cls_int <= len(labels) else f"Class {cls_int}",
                    "pixel_count": pixel_count,
                    "area_m2": area_m2,
                    "area_ha": area_ha
                }
        
        return rgb_image, pred_label, area_stats, cmap
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None, None, None, None

def main():
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Pixel size input
        pixel_size = st.number_input(
            "📏 Pixel Size (meters)", 
            value=30, 
            min_value=1, 
            max_value=1000,
            help="Resolution of each pixel in meters"
        )
        st.session_state['pixel_size'] = pixel_size
        
        st.markdown("---")
        
        # Model status
        with st.spinner("🔄 Loading Model..."):
            model = load_classification_model()
            lc_df, cmap, legend, labels = load_landcover_metadata()
        
        if model is not None:
            st.success("✅ Model Ready!")
            st.info(f"🔧 Input Shape: {model.input_shape}")
        else:
            st.error("❌ Model Failed to Load")
            st.stop()
        
        st.markdown("---")
        
        # Land cover classes with enhanced styling
        st.markdown("### 🗂️ Land Cover Classes")
        for i, (value, label, color) in enumerate(zip(lc_df['values'], lc_df['label'], lc_df['palette'])):
            st.markdown(f"""
            <div class="class-legend">
                <div class="class-color" style="background-color: {color};"></div>
                <span><strong>{value}.</strong> {label}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        st.markdown("""
        <div class="info-card">
            <h3>📤 Upload Satellite Image</h3>
            <p>Upload a GeoTIFF satellite image for advanced land cover classification using deep learning technology</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type=['tif', 'tiff'],
            help="Supported formats: GeoTIFF (.tif, .tiff)",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.markdown(f"""
            <div class="success-message">
                ✅ <strong>File uploaded successfully:</strong> {uploaded_file.name}
                <br>📊 File size: {uploaded_file.size / 1024 / 1024:.2f} MB
                <br>🕒 Ready for processing
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🚀 Start Classification", type="primary", use_container_width=True):
                with st.spinner("🤖 AI model is analyzing your satellite image..."):
                    progress_bar = st.progress(0)
                    
                    # Simulate processing steps with enhanced messages
                    progress_bar.progress(20)
                    st.write("📡 Reading satellite data and preprocessing...")
                    
                    progress_bar.progress(40)
                    st.write("🧠 Running deep learning inference...")
                    
                    rgb_image, pred_label, area_stats, cmap_used = process_image(
                        uploaded_file, model, cmap, legend, labels
                    )
                    
                    progress_bar.progress(80)
                    st.write("📊 Calculating area statistics and metrics...")
                    
                    progress_bar.progress(100)
                    st.write("✅ Analysis completed successfully!")
                
                if rgb_image is not None and pred_label is not None:
                    st.markdown("""
                    <div class="success-message">
                        🎉 <strong>Classification completed successfully!</strong>
                        <br>📈 Your results are ready for analysis
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualization section
                    st.markdown("### 🗺️ Results Visualization")
                    
                    # Try Plotly first, fallback to matplotlib
                    try:
                        fig = create_plotly_visualization(rgb_image, pred_label, labels, lc_df['palette'])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        # st.warning("⚠️ Using fallback visualization")
                        fig = create_matplotlib_visualization(rgb_image, pred_label, cmap_used, legend)
                        st.pyplot(fig)
                    
                    # Area statistics section
                    st.markdown("### 📊 Comprehensive Analysis Results")
                    
                    if area_stats:
                        # Enhanced metrics row
                        total_area_ha = sum(info['area_ha'] for info in area_stats.values())
                        total_classes = len(area_stats)
                        largest_class = max(area_stats.items(), key=lambda x: x[1]['area_ha'])
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{total_area_ha:.1f}</div>
                                <div class="metric-label">📐 Total Area (hectares)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{total_classes}</div>
                                <div class="metric-label">🏷️ Land Cover Classes</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col3:
                            dominant_percentage = (largest_class[1]['area_ha'] / total_area_ha) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{dominant_percentage:.1f}%</div>
                                <div class="metric-label">🎯 Dominant Class</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Enhanced statistics table
                        st.markdown("#### 📋 Detailed Classification Statistics")
                        stats_data = []
                        for cls, info in area_stats.items():
                            percentage = (info['area_ha'] / total_area_ha) * 100 if total_area_ha > 0 else 0
                            stats_data.append({
                                "🏷️ Land Cover Type": info['label'],
                                "📊 Pixel Count": f"{info['pixel_count']:,}",
                                "📐 Area (hectares)": f"{info['area_ha']:.2f}",
                                # "📈 Coverage (%)": f"{percentage:.1f}%",
                                "🗺️ Area (m²)": f"{info['area_m2']:,.0f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(
                            stats_df, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "📈 Coverage (%)": st.column_config.ProgressColumn(
                                    "📈 Coverage (%)",
                                    help="Percentage of total area",
                                    min_value=0,
                                    max_value=100,
                                    format="%.1f%%"
                                )
                            }
                        )

                        st.markdown("---")

                        # Enhanced pie chart section
                        st.markdown("#### 📈 Land Cover Distribution Visualization")
                        chart_fig = create_area_chart(area_stats, labels, lc_df['palette'])
                        if chart_fig:
                            st.plotly_chart(chart_fig, use_container_width=True)
                        
                        # Enhanced download section
                        st.markdown("### 💾 Export Analysis Results")
                        
                        col_csv, col_json, col_summary = st.columns(3)
                        
                        with col_csv:
                            csv = stats_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download CSV Report",
                                data=csv,
                                file_name=f"landcover_analysis_{uploaded_file.name.split('.')[0]}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Download detailed statistics as CSV"
                            )
                        
                        with col_json:
                            json_data = json.dumps(area_stats, indent=2)
                            st.download_button(
                                label="🔧 Download JSON Data",
                                data=json_data,
                                file_name=f"landcover_data_{uploaded_file.name.split('.')[0]}.json",
                                mime="application/json",
                                use_container_width=True,
                                help="Download raw data as JSON"
                            )
                        
                        with col_summary:
                            # Generate summary report
                            summary_report = f"""
LAND COVER ANALYSIS SUMMARY
===========================

File: {uploaded_file.name}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Pixel Resolution: {pixel_size}m

OVERVIEW:
- Total Area Analyzed: {total_area_ha:.2f} hectares
- Number of Land Cover Classes: {total_classes}
- Dominant Land Cover: {largest_class[1]['label']} ({dominant_percentage:.1f}%)

DETAILED BREAKDOWN:
"""
                            for cls, info in sorted(area_stats.items(), key=lambda x: x[1]['area_ha'], reverse=True):
                                percentage = (info['area_ha'] / total_area_ha) * 100
                                summary_report += f"- {info['label']}: {info['area_ha']:.2f} ha ({percentage:.1f}%)\n"
                            
                            st.download_button(
                                label="📄 Download Summary",
                                data=summary_report,
                                file_name=f"summary_report_{uploaded_file.name.split('.')[0]}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                help="Download executive summary"
                            )
                    
                    else:
                        st.markdown("""
                        <div class="info-card" style="border-left: 4px solid #ffa500;">
                            <h4>⚠️ Analysis Notice</h4>
                            <p>No valid land cover classes were detected in the processed image. This could be due to:</p>
                            <ul>
                                <li>Image quality or resolution issues</li>
                                <li>Incompatible spectral bands</li>
                                <li>Model training domain mismatch</li>
                            </ul>
                            <p>Please try with a different image or check the input requirements.</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        # Enhanced information panels
        st.markdown("""
        <div class="info-card">
            <h3>💡 How It Works</h3>
            <ol style="line-height: 1.8;">
                <li><strong>🗂️ Upload</strong> your GeoTIFF satellite image</li>
                <li><strong>⚙️ Configure</strong> pixel resolution in sidebar</li>
                <li><strong>🚀 Process</strong> with AI classification</li>
                <li><strong>📊 Analyze</strong> interactive results</li>
                <li><strong>💾 Export</strong> comprehensive reports</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📋 Technical Requirements</h3>
            <ul style="line-height: 1.8;">
                <li>🗂️ <strong>Format:</strong> GeoTIFF (.tif, .tiff)</li>
                <li>📊 <strong>Bands:</strong> Multi-spectral imagery</li>
                <li>📐 <strong>Resolution:</strong> Compatible dimensions</li>
                <li>🌍 <strong>Source:</strong> Satellite/aerial imagery</li>
                <li>💾 <strong>Size:</strong> Optimized for processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>✨ Advanced Features</h3>
            <ul style="line-height: 1.8;">
                <li>🤖 <strong>AI-Powered:</strong> Deep learning classification</li>
                <li>🎯 <strong>Interactive:</strong> Plotly visualizations</li>
                <li>📊 <strong>Comprehensive:</strong> Detailed statistics</li>
                <li>💾 <strong>Multi-format:</strong> CSV, JSON, Summary exports</li>
                <li>📱 <strong>Responsive:</strong> Modern, mobile-friendly UI</li>
                <li>🔄 <strong>Real-time:</strong> Progress tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Use Cases</h3>
            <ul style="line-height: 1.8;">
                <li>🌿 <strong>Environmental:</strong> Forest monitoring</li>
                <li>🚜 <strong>Agriculture:</strong> Crop mapping</li>
                <li>🏙️ <strong>Urban:</strong> Land use planning</li>
                <li>💧 <strong>Water:</strong> Resource management</li>
                <li>📈 <strong>Research:</strong> Change detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("""
    <div class="footer">
        <div style="text-align: center; color: #4a5568;">
            <h4 style="margin-bottom: 1rem; color: #2d3748;">🛰️ Land Cover Classification Platform</h4>
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                <strong>Powered by Deep Learning & Advanced Analytics</strong>
            </p>
            <p style="opacity: 0.8; margin-bottom: 1rem;">
                Built for precision satellite imagery analysis and environmental monitoring
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <span>🧠 <strong>AI-Driven</strong></span>
                <span>📊 <strong>Data-Rich</strong></span>
                <span>🌍 <strong>Earth-Focused</strong></span>
            </div>
            <p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.7;">
                Developed by <strong>Aditya Dharma</strong> • Version 2.0
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()