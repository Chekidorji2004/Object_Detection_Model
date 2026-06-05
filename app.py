import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

import torch
torch.classes.__path__ = []

# Page configuration
st.set_page_config(
    page_title="Bhutan Wildlife Guardian - AI Species Detection",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, responsive UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #2c3e50 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* About section */
    .about-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #27ae60;
    }
    
    /* Species card styling */
    .species-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        border: 1px solid #e0e0e0;
        margin: 0.5rem;
    }
    
    .species-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .species-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .species-name {
        font-weight: bold;
        font-size: 1rem;
        color: #333;
    }
    
    .species-scientific {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    
    /* Detection results styling */
    .detection-card {
        background: linear-gradient(135deg, #2c3e50 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .status-CR { background-color: #dc3545; color: white; }
    .status-EN { background-color: #fd7e14; color: white; }
    .status-VU { background-color: #ffc107; color: #333; }
    .status-NT { background-color: #20c997; color: white; }
    
    .threat-item, .action-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    
    /* Upload section */
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    /* Confidence meter */
    .confidence-meter {
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        transition: width 0.3s ease;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2c3e50 0%, #27ae60 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        color: white;
    }
    
    hr {
        margin: 2rem 0;
    }
    
    /* Input method buttons */
    .input-method-btn {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Updated Bhutan-focused species database
SPECIES_DATA = {
    "Red_Panda": {
        "icon": "🐼",
        "scientific": "Ailurus fulgens",
        "dzongkha_name": "ཨ་ཅོ་དོང་ཀར།",
        "conservation_status": "EN",
        "status_text": "Endangered",
        "population": "Approximately 300-400 in Bhutan",
        "description": "The Red Panda is one of Bhutan’s flagship conservation species found in temperate forests with dense bamboo understories. It is mainly distributed in the eastern and northern Himalayan regions of Bhutan and plays an important role in maintaining healthy forest ecosystems.",
        "threats": [
            "Habitat fragmentation caused by road construction and hydropower projects",
            "Loss of bamboo forests due to climate change",
            "Poaching and illegal wildlife trade",
            "Disturbance from livestock grazing and forest resource extraction",
            "Forest fires and human encroachment"
        ],
        "actions": [
            "Support Bhutan’s biological corridor conservation efforts",
            "Promote community forestry and sustainable land use",
            "Participate in red panda awareness programs",
            "Encourage eco-tourism initiatives supporting local communities",
            "Report illegal wildlife activities to forestry officials"
        ],
        "fun_fact": "Bhutan is considered one of the most important strongholds for red pandas in the Eastern Himalayas.",
        "habitat_elevation": "2,100 - 4,300m",
        "protected_areas": [
            "Jigme Dorji National Park",
            "Wangchuck Centennial National Park",
            "Bumdeling Wildlife Sanctuary"
        ]
    },

    "Snow_Leopard": {
        "icon": "🐆",
        "scientific": "Panthera uncia",
        "dzongkha_name": "གངས་གཟིག།",
        "conservation_status": "VU",
        "status_text": "Vulnerable",
        "population": "Estimated 100-200 in Bhutan",
        "description": "The Snow Leopard inhabits Bhutan’s high alpine mountains and remote northern regions. Known as the ‘Ghost of the Mountains,’ it is perfectly adapted to cold Himalayan environments and serves as an indicator of healthy mountain ecosystems.",
        "threats": [
            "Retaliatory killing due to livestock predation",
            "Decline in natural prey species",
            "Climate change affecting alpine habitats",
            "Infrastructure development in fragile mountain ecosystems",
            "Poaching and illegal wildlife trade"
        ],
        "actions": [
            "Support livestock insurance and predator-proof corrals",
            "Strengthen community-based conservation programs",
            "Support anti-poaching patrols in northern Bhutan",
            "Promote sustainable mountain tourism",
            "Participate in snow leopard monitoring projects"
        ],
        "fun_fact": "Snow leopards in Bhutan can survive temperatures below -30°C in the high Himalayas.",
        "habitat_elevation": "3,500 - 5,500m",
        "protected_areas": [
            "Jigme Dorji National Park",
            "Wangchuck Centennial National Park"
        ]
    },

    "Bengal_Tiger": {
        "icon": "🐯",
        "scientific": "Panthera tigris tigris",
        "dzongkha_name": "སྟག།",
        "conservation_status": "EN",
        "status_text": "Endangered",
        "population": "Approximately 130 tigers in Bhutan",
        "description": "Bhutan is the first country in the world to record tigers living as high as 4,000 meters above sea level. Bengal Tigers are found across Bhutan’s subtropical forests to alpine regions, demonstrating the country’s exceptional ecological connectivity.",
        "threats": [
            "Poaching for illegal wildlife trade",
            "Habitat fragmentation from development projects",
            "Human-wildlife conflict near settlements",
            "Decline of prey populations",
            "Climate change impacting forest ecosystems"
        ],
        "actions": [
            "Support tiger conservation and monitoring programs",
            "Protect biological corridors connecting tiger habitats",
            "Promote community-based conservation awareness",
            "Support anti-poaching initiatives",
            "Encourage sustainable development practices"
        ],
        "fun_fact": "Bhutan’s tigers are known to migrate vertically from tropical forests to alpine mountains.",
        "habitat_elevation": "150 - 4,000m",
        "protected_areas": [
            "Royal Manas National Park",
            "Jigme Singye Wangchuck National Park",
            "Jigme Dorji National Park"
        ]
    },

    "Asian_Elephant": {
        "icon": "🐘",
        "scientific": "Elephas maximus",
        "dzongkha_name": "གླང་ཆེན།",
        "conservation_status": "EN",
        "status_text": "Endangered",
        "population": "Approximately 600-800 in Bhutan",
        "description": "Asian Elephants in Bhutan are mainly found in the southern foothills and subtropical forests. They are highly social animals and play a major ecological role by dispersing seeds and creating forest clearings.",
        "threats": [
            "Human-elephant conflict in farming communities",
            "Habitat loss from agricultural expansion",
            "Fragmentation of elephant migration routes",
            "Poaching for ivory",
            "Increasing infrastructure development"
        ],
        "actions": [
            "Support elephant corridor conservation",
            "Promote human-elephant conflict mitigation programs",
            "Install community-based early warning systems",
            "Support wildlife-friendly farming practices",
            "Participate in conservation awareness campaigns"
        ],
        "fun_fact": "Asian elephants can travel long distances across Bhutan’s biological corridors in search of food and water.",
        "habitat_elevation": "100 - 2,000m",
        "protected_areas": [
            "Royal Manas National Park",
            "Jigme Singye Wangchuck National Park",
            "Phibsoo Wildlife Sanctuary"
        ]
    },

    "Takin": {
        "icon": "🦬",
        "scientific": "Budorcas taxicolor whitei",
        "dzongkha_name": "འབྲོང་གྱིམ་རྩེ།",
        "conservation_status": "VU",
        "status_text": "Vulnerable",
        "population": "Stable but locally threatened in Bhutan",
        "description": "The Takin is Bhutan’s national animal and holds deep cultural and spiritual significance. It inhabits alpine meadows and temperate forests and is especially associated with the legend of Drukpa Kunley, the Divine Madman.",
        "threats": [
            "Habitat degradation from livestock grazing",
            "Climate change affecting alpine ecosystems",
            "Disease transmission from domestic cattle",
            "Disturbance from infrastructure development",
            "Illegal hunting in remote areas"
        ],
        "actions": [
            "Protect alpine meadows and migration routes",
            "Promote sustainable livestock management",
            "Support wildlife disease monitoring",
            "Encourage conservation education programs",
            "Strengthen protected area management"
        ],
        "fun_fact": "The Takin is believed to have been created by the Divine Madman by joining the head of a goat to the body of a cow.",
        "habitat_elevation": "1,500 - 4,500m",
        "protected_areas": [
            "Jigme Dorji National Park",
            "Wangchuck Centennial National Park",
            "Jigme Singye Wangchuck National Park"
        ]
    },

    "Clouded_Leopard": {
        "icon": "🐅",
        "scientific": "Neofelis nebulosa",
        "dzongkha_name": "སྤྲིན་གཟིག།",
        "conservation_status": "VU",
        "status_text": "Vulnerable",
        "population": "Rare and elusive in Bhutan",
        "description": "The Clouded Leopard inhabits Bhutan’s dense subtropical and temperate forests. It is one of the least studied wild cats in Bhutan and is known for its remarkable climbing abilities and beautiful cloud-shaped markings.",
        "threats": [
            "Deforestation and habitat fragmentation",
            "Poaching for fur and illegal trade",
            "Decline in prey populations",
            "Road construction through forest habitats",
            "Lack of ecological research data"
        ],
        "actions": [
            "Support forest conservation programs",
            "Expand camera trap monitoring projects",
            "Protect wildlife corridors",
            "Promote scientific research on elusive species",
            "Raise awareness about illegal wildlife trade"
        ],
        "fun_fact": "Clouded leopards can climb down trees headfirst thanks to their powerful limbs and flexible ankles.",
        "habitat_elevation": "200 - 3,500m",
        "protected_areas": [
            "Royal Manas National Park",
            "Jigme Singye Wangchuck National Park",
            "Phibsoo Wildlife Sanctuary"
        ]
    },

    "Musk_Deer": {
        "icon": "🦌",
        "scientific": "Moschus chrysogaster",
        "dzongkha_name": "གླ་བྷ།",
        "conservation_status": "EN",
        "status_text": "Endangered",
        "population": "Declining in Bhutan",
        "description": "The Himalayan Musk Deer is a shy alpine species found in Bhutan’s northern forests and shrublands. Males produce musk used in perfumes and traditional medicine, making the species highly vulnerable to illegal hunting.",
        "threats": [
            "Poaching for musk pods",
            "Habitat degradation in alpine regions",
            "Climate change impacting mountain vegetation",
            "Accidental trapping and snaring",
            "Limited population monitoring"
        ],
        "actions": [
            "Strengthen anti-poaching enforcement",
            "Promote alternative livelihoods for local communities",
            "Support conservation research and monitoring",
            "Protect alpine forest habitats",
            "Raise awareness against illegal wildlife products"
        ],
        "fun_fact": "Unlike most deer species, male musk deer have long saber-like canine teeth instead of antlers.",
        "habitat_elevation": "2,500 - 4,500m",
        "protected_areas": [
            "Jigme Dorji National Park",
            "Wangchuck Centennial National Park",
            "Bumdeling Wildlife Sanctuary"
        ]
    }
}

SPECIES_LIST = list(SPECIES_DATA.keys())

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')  # Make sure best.pt is in the same directory
        return model
    except Exception as e:
        st.error(f"⚠️ Model loading error: {str(e)}")
        st.info("Please ensure 'best.pt' file is in the same directory as this application")
        return None

def detect_species(image, model, confidence_threshold=0.8):
    """Run YOLO detection with confidence threshold"""
    try:
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        results = model(image_np)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            # Get highest confidence detection
            best_idx = np.argmax(confidences)
            confidence = confidences[best_idx]
            
            # Check confidence threshold
            if confidence >= confidence_threshold:
                class_id = class_ids[best_idx]
                species_name = results[0].names[class_id]
                box = boxes[best_idx].xyxy[0].cpu().numpy()
                
                # Check if species is in our database
                if species_name in SPECIES_LIST:
                    return species_name, confidence, True, box
                else:
                    return species_name, confidence, False, None
            else:
                return None, confidence, False, None
        else:
            return None, 0, False, None
            
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None, 0, False, None

def draw_bounding_box(image, box, species_name, confidence):
    """Draw bounding box on image"""
    img_copy = np.array(image.copy())
    x1, y1, x2, y2 = map(int, box)
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (39, 174, 96), 3)
    
    # Add label background
    label = f"{species_name} ({confidence:.1%})"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), 
                  (x1 + label_size[0] + 10, y1), (39, 174, 96), -1)
    cv2.putText(img_copy, label, (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_copy

# ==================== MAIN APPLICATION ====================

# Header Section
st.markdown("""
<div class='header'>
    <h1>🏔️ Bhutan Wildlife Guardian</h1>
    <p>AI-Powered Species Detection and Conservation Platform</p>
    <p style='font-size: 0.9rem; opacity: 0.9;'>Protecting Bhutan's Natural Heritage with YOLOv11</p>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<div class='about-section'>
    <h3>About This Platform</h3>
    <p>The Bhutan Wildlife Guardian is an AI-powered conservation tool designed to help identify and protect Bhutan's most iconic wildlife species. Using state-of-the-art YOLOv11 deep learning technology, this platform can:</p>
    <ul>
        <li><b>Detect 7 Key Species:</b> Red Panda, Snow Leopard, Bengal Tiger, Asian Elephant, Takin, Clouded Leopard, and Black Musk Deer</li>
        <li><b>Provide Conservation Information:</b> Learn about species status, threats, and how you can help</li>
        <li><b>Support Conservation:</b> Educational content to promote wildlife protection</li>
    </ul>
    <p><i>Note: Detection requires 80%+ confidence for accurate identification. Upload clear, well-lit images for best results!</i></p>
</div>
""", unsafe_allow_html=True)

# Species Grid Section
st.markdown("---")
st.subheader("Our Target Species")

# Create responsive species grid with 4 cards per row
cols = st.columns(4)
for idx, species in enumerate(SPECIES_LIST):
    col_idx = idx % 4
    data = SPECIES_DATA[species]
    with cols[col_idx]:
        st.markdown(f"""
        <div class='species-card'>
            <div class='species-icon'>{data['icon']}</div>
            <div class='species-name'>{species}</div>
            <div class='species-scientific'>{data['scientific']}</div>
            <div style='margin-top: 0.5rem;'>
                <span class='status-badge status-{data['conservation_status']}'>{data['status_text']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Image Upload Section
st.markdown("---")
st.subheader("Species Detection")

# Initialize session state for input method
if 'input_method' not in st.session_state:
    st.session_state.input_method = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None

# Create two columns for input method selection
col1, col2 = st.columns(2)

with col1:
    upload_button = st.button("📁 Upload Image", use_container_width=True, key="upload_btn")
    if upload_button:
        st.session_state.input_method = "upload"
        st.session_state.camera_image = None  # Clear camera image

with col2:
    camera_button = st.button("📷 Take Photo", use_container_width=True, key="camera_btn")
    if camera_button:
        st.session_state.input_method = "camera"
        st.session_state.uploaded_image = None  # Clear uploaded image

st.markdown("---")

# Show the appropriate input widget based on selection
if st.session_state.input_method == "upload":
    st.markdown("### 📁 Upload Image from Computer")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
        help="Upload a clear image of Bhutanese wildlife",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)
        
elif st.session_state.input_method == "camera":
    st.markdown("### 📷 Take a Photo")
    st.info("📸 Allow camera access when prompted")
    camera_image = st.camera_input("Capture photo...", key="camera_input")
    
    if camera_image is not None:
        st.session_state.camera_image = Image.open(camera_image)
        st.image(st.session_state.camera_image, caption="Captured Image", use_container_width=True)
        
else:
    # No method selected yet
    st.info("👆 Please select either 'Upload Image' or 'Take Photo' to begin")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin: 1rem 0;'>
        <p style='font-size: 1.2rem;'>📸 Choose an input method above</p>
        <p>Upload an existing image or take a new photo using your camera</p>
    </div>
    """, unsafe_allow_html=True)

# Get the current image
current_image = None
if st.session_state.input_method == "upload" and st.session_state.uploaded_image is not None:
    current_image = st.session_state.uploaded_image
elif st.session_state.input_method == "camera" and st.session_state.camera_image is not None:
    current_image = st.session_state.camera_image

# Detection button and logic
if current_image is not None:
    st.markdown("---")
    
    # Detection button
    if st.button("🔍 Detect Species", type="primary", use_container_width=True):
        # Load model
        model = load_model()
        
        if model is not None:
            with st.spinner("Analyzing image with YOLOv11..."):
                # Run detection with 80% confidence threshold
                species_name, confidence, is_target, box = detect_species(current_image, model, confidence_threshold=0.8)
                
                if species_name is not None and is_target and confidence >= 0.8:
                    # Draw bounding box
                    image_with_box = draw_bounding_box(current_image, box, species_name, confidence)
                    
                    # Display detection result
                    st.markdown("---")
                    st.markdown("## ✅ Detection Result")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.image(image_with_box, caption=f"Detected: {species_name}", use_container_width=True)
                    
                    with col2:
                        data = SPECIES_DATA[species_name]
                        st.markdown(f"""
                        <div class='detection-card'>
                            <h2>{data['icon']} {species_name}</h2>
                            <p><b>Confidence Score:</b></p>
                            <div class='confidence-meter'>
                                <div class='confidence-fill' style='width: {confidence*100}%;'></div>
                            </div>
                            <p style='font-size: 1.5rem; font-weight: bold;'>{confidence:.1%}</p>
                            <p><b>Scientific Name:</b> <i>{data['scientific']}</i></p>
                            <p><b>Dzongkha Name:</b> {data['dzongkha_name']}</p>
                            <p><b>Conservation Status:</b> <span class='status-badge status-{data['conservation_status']}'>{data['status_text']}</span></p>
                            <p><b>Population:</b> {data['population']} individuals</p>
                            <p><b>Habitat Elevation:</b> {data['habitat_elevation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Species Information
                    st.markdown("---")
                    st.subheader(f"About {species_name}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='info-card'>
                            <h4>Description</h4>
                            <p>{data['description']}</p>
                            <h4>Fun Fact</h4>
                            <p>{data['fun_fact']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='info-card'>
                            <h4>Major Threats</h4>
                        """, unsafe_allow_html=True)
                        for threat in data['threats']:
                            st.markdown(f"<div class='threat-item'>• {threat}</div>", unsafe_allow_html=True)
                        
                        st.markdown("""
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Conservation Actions
                    st.markdown("""
                    <div class='info-card'>
                        <h4>How You Can Help</h4>
                    """, unsafe_allow_html=True)
                    for action in data['actions']:
                        st.markdown(f"<div class='action-item'>✅ {action}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Protected areas
                    st.markdown("---")
                    st.subheader(f"Protected Areas Hosting {species_name}")
                    for pa in data['protected_areas']:
                        st.success(f"📍 {pa}")
                    
                    # Reset button
                    st.markdown("---")
                    if st.button("Detect Another Image", use_container_width=True):
                        st.session_state.input_method = None
                        st.session_state.uploaded_image = None
                        st.session_state.camera_image = None
                        st.rerun()
                    
                elif species_name is not None and not is_target:
                    st.warning(f"⚠️ Detection Results")
                    st.info(f"""
                    **Detected: {species_name}** (Confidence: {confidence:.1%})
                    
                    This species is not in our target list. Our system can only detect:
                    {', '.join(SPECIES_LIST)}
                    
                    Please upload an image of one of these species for detailed information.
                    """)
                    
                elif confidence < 0.8:
                    st.warning("⚠️ Low Confidence Detection")
                    st.info(f"""
                    **Confidence Score: {confidence:.1%}**
                    
                    The detection confidence is below the 80% threshold. This could be due to:
                    - Poor image quality or lighting
                    - Animal is too small or blurry
                    - Multiple animals in the image
                    
                    **Please try:**
                    - Uploading a clearer image
                    - Ensuring good lighting
                    - Centering the animal in frame
                    """)
                    
                else:
                    st.error("❌ No Species Detected")
                    st.info("""
                    No target species were detected in this image. This could be because:
                    - No animal is present in the image
                    - The animal is not one of our 7 target species
                    - The image quality is too low
                    
                    **Target species include:**
                    Red Panda, Snow Leopard, Bengal Tiger, Asian Elephant, Takin, Clouded Leopard, Black Musk Deer
                    
                    Try uploading a clearer image for better results!
                    """)
        else:
            st.error("Model not loaded. Please check your 'best.pt' file.")

# Reset button when no image selected but method was chosen
if st.session_state.input_method in ["upload", "camera"] and current_image is None:
    if st.button("🔄 Reset and Try Again", use_container_width=True):
        st.session_state.input_method = None
        st.session_state.uploaded_image = None
        st.session_state.camera_image = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #2c3e50 0%, #27ae60 100%); border-radius: 15px; color: white; margin-top: 2rem;'>
    <h4>🌿 Protecting Bhutan's Wildlife for Future Generations</h4>
    <p>Powered by YOLOv11 | AI for Conservation | Bhutan Wildlife Guardian</p>
    <p style='font-size: 0.8rem; opacity: 0.9;'>© 2026 - Supporting Bhutan's Conservation Efforts</p>
</div>
""", unsafe_allow_html=True)
