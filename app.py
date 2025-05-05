import streamlit as st
import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
from transformers import pipeline
from PIL import Image
import tempfile

# Page configuration
st.set_page_config(page_title="Multi-Function AI App", page_icon="🧠", layout="wide")

# Apply custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0f9d58;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and introduction
st.markdown(
    "<h1 class='main-header'>Multi-Function AI App</h1>", unsafe_allow_html=True
)
st.markdown(
    "This application combines face recognition, face analysis, and text summarization features."
)


# Create necessary directories
def create_directories():
    dataset_path = "./face_dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    return dataset_path


# Initialize session state
if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
if "face_cascade" not in st.session_state:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    st.session_state.face_cascade = cv2.CascadeClassifier(cascade_path)


# KNN for face recognition
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


# Helper function to convert NumPy types for JSON serialization
def convert(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj


# Helper function to get faces and store faces
def get_faces(frame, face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    return faces, gray_frame


# Main sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the App Mode",
    ["Home", "Face Recognition", "Face Analysis", "Text Summarization"],
)

# Home page
if app_mode == "Home":
    st.markdown(
        "<h2 class='sub-header'>Welcome to the Multi-Function AI App!</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    This application offers three main functionalities:
    
    1. **Face Recognition**: Register new faces and recognize existing ones
    2. **Face Analysis**: Analyze faces for age, gender, emotion, and ethnicity
    3. **Text Summarization**: Generate concise summaries of longer texts
    
    Select a function from the sidebar to get started!
    """
    )

    # Display images for each functionality
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            "https://media.istockphoto.com/id/1341619309/photo/facial-recognition-system-concept.jpg?s=612x612&w=0&k=20&c=p8Ym8PwXJMpYdJnP9_9x5oBY5GMC1jwhLNEeVhaF2AI=",
            caption="Face Recognition",
        )
    with col2:
        st.image(
            "https://t4.ftcdn.net/jpg/03/56/17/75/360_F_356177598_4G1MIcLllK5xQ9jbJ2DXiTS7iUgAlGGV.jpg",
            caption="Face Analysis",
        )
    with col3:
        st.image(
            "https://cdn.pixabay.com/photo/2016/09/10/17/18/book-1659717_1280.jpg",
            caption="Text Summarization",
        )

# Face Recognition functionality
elif app_mode == "Face Recognition":
    st.markdown("<h2 class='sub-header'>Face Recognition</h2>", unsafe_allow_html=True)

    face_recognition_mode = st.radio(
        "Choose an operation", ["Register New Face", "Recognize Faces"]
    )

    if face_recognition_mode == "Register New Face":
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Register a New Face")

        file_name = st.text_input("Enter name for the face data:")

        if st.button("Start Face Capture") and file_name:
            dataset_path = create_directories()
            stframe = st.empty()
            st_status = st.empty()

            face_data = []
            cap = cv2.VideoCapture(0)

            # Collect samples
            for i in range(30):  # Collect 30 samples
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not access the camera.")
                    break

                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faces, gray_frame = get_faces(frame, st.session_state.face_cascade)

                if len(faces) > 0:
                    # Sort faces by area (to pick the largest face)
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

                    # Take the largest face
                    for face in faces[:1]:
                        x, y, w, h = face
                        # Draw rectangle
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Extract face section with padding
                        offset = 10
                        face_section = frame[
                            y - offset : y + h + offset, x - offset : x + w + offset
                        ]

                        try:
                            face_section = cv2.resize(face_section, (100, 100))
                            face_data.append(face_section)
                            st_status.text(f"Collected {len(face_data)} samples")
                        except:
                            continue

                # Display the frame
                stframe.image(frame_rgb, channels="RGB", caption="Live Feed")
                time.sleep(0.2)  # Short delay

            cap.release()

            if face_data:
                # Save face data
                face_data = np.asarray(face_data)
                face_data = face_data.reshape(
                    (face_data.shape[0], -1)
                )  # Flatten the data
                np.save(os.path.join(dataset_path, file_name + ".npy"), face_data)
                st.success(f"Face data saved for {file_name}!")
            else:
                st.error("No face data collected. Please try again.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif face_recognition_mode == "Recognize Faces":
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Recognize Faces")

        if st.button("Start Recognition"):
            # Load face data
            dataset_path = create_directories()
            face_data = []
            labels = []
            class_id = 0
            names = {}

            # Check if there's any face data available
            if not os.listdir(dataset_path):
                st.error("No face data found. Please register faces first.")
            else:
                for fx in os.listdir(dataset_path):
                    if fx.endswith(".npy"):
                        names[class_id] = fx[:-4]
                        data_item = np.load(os.path.join(dataset_path, fx))
                        face_data.append(data_item)
                        target = class_id * np.ones((data_item.shape[0],))
                        labels.append(target)
                        class_id += 1

                if not face_data:
                    st.error("No face data loaded. Please check the dataset.")
                else:
                    face_dataset = np.concatenate(face_data, axis=0)
                    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
                    trainset = np.concatenate((face_dataset, face_labels), axis=1)

                    # Start recognition
                    stframe = st.empty()
                    st_status = st.empty()

                    cap = cv2.VideoCapture(0)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Recognition loop
                    recognition_active = True
                    st_status.text(
                        "Recognition active. Click 'Stop Recognition' to end."
                    )

                    stop_button = st.button("Stop Recognition")

                    while recognition_active and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Could not access the camera.")
                            break

                        # Convert to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Detect faces
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = st.session_state.face_cascade.detectMultiScale(
                            gray, 1.3, 5
                        )

                        for face in faces:
                            x, y, w, h = face
                            offset = 10
                            face_section = frame[
                                y - offset : y + h + offset, x - offset : x + w + offset
                            ]

                            if face_section.size == 0:
                                continue

                            try:
                                face_section = cv2.resize(face_section, (100, 100))
                                # Flatten for prediction
                                out = knn(trainset, face_section.flatten())
                                name = names[int(out)]

                                # Draw name and rectangle
                                cv2.putText(
                                    frame_rgb,
                                    name,
                                    (x, y - 10),
                                    font,
                                    1,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                                cv2.rectangle(
                                    frame_rgb,
                                    (x, y),
                                    (x + w, y + h),
                                    (255, 255, 255),
                                    2,
                                )
                            except:
                                continue

                        # Display the frame
                        stframe.image(frame_rgb, channels="RGB", caption="Recognition")

                        # Check for stop button
                        if stop_button:
                            recognition_active = False

                        time.sleep(0.1)  # Short delay

                    cap.release()
                    st_status.text("Recognition stopped.")

        st.markdown("</div>", unsafe_allow_html=True)

# Face Analysis functionality
elif app_mode == "Face Analysis":
    st.markdown("<h2 class='sub-header'>Face Analysis</h2>", unsafe_allow_html=True)
    st.write("Analyze faces for age, gender, emotion, and ethnicity.")

    # Check and install deepface if not already installed
    try:
        from deepface import DeepFace

        deepface_loaded = True
    except ImportError:
        st.warning(
            "DeepFace library is not installed. Please install it by running: pip install deepface"
        )
        st.info("You may also need to restart the app after installation.")
        deepface_loaded = False

    if deepface_loaded:
        analysis_mode = st.radio(
            "Choose input method:", ["Upload Image", "Live Webcam"]
        )

        if analysis_mode == "Upload Image":
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as tmp_file:
                    fp = tmp_file.name
                    tmp_file.write(uploaded_file.getvalue())

                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Analyze button
                if st.button("Analyze Face"):
                    with st.spinner("Analyzing face, please wait..."):
                        try:
                            result = DeepFace.analyze(
                                img_path=fp,
                                actions=["age", "gender", "emotion", "race"],
                                enforce_detection=False,
                            )

                            if isinstance(result, list):
                                result = result[0]

                            analyzed_info = {
                                "Age": convert(result.get("age")),
                                "Gender": {
                                    k: convert(v)
                                    for k, v in result.get("gender", {}).items()
                                },
                                "Emotion": result.get("emotion"),
                                "Race": result.get("race"),
                            }

                            # Display results
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Age")
                                st.info(f"{analyzed_info['Age']} years")

                                st.subheader("Gender")
                                gender_data = analyzed_info["Gender"]
                                dominant_gender = max(gender_data, key=gender_data.get)
                                st.info(
                                    f"{dominant_gender}: {gender_data[dominant_gender]:.2f}%"
                                )

                            with col2:
                                st.subheader("Primary Emotion")
                                emotion_data = analyzed_info["Emotion"]
                                dominant_emotion = max(
                                    emotion_data, key=emotion_data.get
                                )
                                st.info(
                                    f"{dominant_emotion.capitalize()}: {emotion_data[dominant_emotion]:.2f}%"
                                )

                                st.subheader("Primary Ethnicity")
                                race_data = analyzed_info["Race"]
                                dominant_race = max(race_data, key=race_data.get)
                                st.info(
                                    f"{dominant_race.capitalize()}: {race_data[dominant_race]:.2f}%"
                                )

                            # Display detailed results
                            with st.expander("See detailed analysis"):
                                st.json(analyzed_info)

                            # Save results to file
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"image_analysis_{timestamp}.json"
                            with open(filename, "w") as f:
                                json.dump(analyzed_info, f, indent=4)

                            st.success(f"Analysis completed and saved to {filename}")

                        except Exception as e:
                            st.error(f"Error analyzing face: {str(e)}")

                        finally:
                            # Clean up the temp file
                            if os.path.exists(fp):
                                os.remove(fp)

            st.markdown("</div>", unsafe_allow_html=True)

        elif analysis_mode == "Live Webcam":
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.write("Capture an image from your webcam for analysis.")

            if st.button("Start Camera"):
                # Initialize camera
                cap = cv2.VideoCapture(0)
                stframe = st.empty()

                # Capture button placeholder
                capture_button_placeholder = st.empty()
                capture_button = capture_button_placeholder.button("Capture Image")

                # Status placeholder
                status_placeholder = st.empty()
                status_placeholder.info(
                    "Webcam active. Press 'Capture Image' to analyze."
                )

                # Result placeholders
                result_placeholder = st.empty()

                # Processed image storage
                captured_image = None

                # Camera loop
                while not capture_button:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("Failed to access camera.")
                        break

                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", caption="Live Feed")

                    # Check if capture button is pressed
                    capture_button = capture_button_placeholder.button(
                        "Capture Image", key="capture_image_unique"
                    )
                    if capture_button:
                        captured_image = frame.copy()
                        break

                    time.sleep(0.1)  # Short delay

                if captured_image is not None:
                    # Save captured image temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jpg"
                    ) as tmp_file:
                        fp = tmp_file.name
                        cv2.imwrite(fp, captured_image)

                    # Display captured image
                    captured_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                    stframe.image(
                        captured_rgb, channels="RGB", caption="Captured Image"
                    )

                    # Analyze face
                    status_placeholder.info("Analyzing captured face, please wait...")

                    try:
                        result = DeepFace.analyze(
                            img_path=fp,
                            actions=["age", "gender", "emotion", "race"],
                            enforce_detection=False,
                        )

                        if isinstance(result, list):
                            result = result[0]

                        analyzed_info = {
                            "Age": convert(result.get("age")),
                            "Gender": {
                                k: convert(v)
                                for k, v in result.get("gender", {}).items()
                            },
                            "Emotion": result.get("emotion"),
                            "Race": result.get("race"),
                        }

                        # Display results
                        status_placeholder.success("Analysis complete!")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Age")
                            st.info(f"{analyzed_info['Age']} years")

                            st.subheader("Gender")
                            gender_data = analyzed_info["Gender"]
                            dominant_gender = max(gender_data, key=gender_data.get)
                            st.info(
                                f"{dominant_gender}: {gender_data[dominant_gender]:.2f}%"
                            )

                        with col2:
                            st.subheader("Primary Emotion")
                            emotion_data = analyzed_info["Emotion"]
                            dominant_emotion = max(emotion_data, key=emotion_data.get)
                            st.info(
                                f"{dominant_emotion.capitalize()}: {emotion_data[dominant_emotion]:.2f}%"
                            )

                            st.subheader("Primary Ethnicity")
                            race_data = analyzed_info["Race"]
                            dominant_race = max(race_data, key=race_data.get)
                            st.info(
                                f"{dominant_race.capitalize()}: {race_data[dominant_race]:.2f}%"
                            )

                        # Display detailed results
                        with st.expander("See detailed analysis"):
                            st.json(analyzed_info)

                        # Save results to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"live_analysis_{timestamp}.json"
                        with open(filename, "w") as f:
                            json.dump(analyzed_info, f, indent=4)

                        st.success(f"Analysis saved to {filename}")

                    except Exception as e:
                        status_placeholder.error(f"Error analyzing face: {str(e)}")

                    finally:
                        # Clean up the temp file
                        if os.path.exists(fp):
                            os.remove(fp)

                # Release the camera
                cap.release()

            st.markdown("</div>", unsafe_allow_html=True)

# Text Summarization functionality
elif app_mode == "Text Summarization":
    st.markdown(
        "<h2 class='sub-header'>Text Summarization</h2>", unsafe_allow_html=True
    )
    st.write("Generate concise summaries of longer texts.")

    st.markdown("<div class='section'>", unsafe_allow_html=True)

    # Initialize the summarizer
    if st.session_state.summarizer is None:
        with st.spinner("Loading summarization model... This may take a moment."):
            try:
                st.session_state.summarizer = pipeline("summarization")
                st.success("Summarization model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load summarization model: {str(e)}")
                st.info(
                    "Make sure you have installed the required packages: pip install transformers torch"
                )

    # Text input
    st.subheader("Enter text to summarize")
    text_to_summarize = st.text_area("Paste your text here:", height=200)

    # Summarization parameters
    st.subheader("Summarization Parameters")
    col1, col2 = st.columns(2)

    with col1:
        max_length = st.slider("Maximum summary length", 30, 500, 130)
    with col2:
        min_length = st.slider("Minimum summary length", 10, 100, 30)

    # Summarize button
    if st.button("Summarize") and text_to_summarize:
        if st.session_state.summarizer is not None:
            with st.spinner("Generating summary..."):
                try:
                    summary = st.session_state.summarizer(
                        text_to_summarize,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                    )

                    # Display the summary
                    st.subheader("Generated Summary")
                    st.info(summary[0]["summary_text"])

                    # Display statistics
                    st.subheader("Statistics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Original Length", len(text_to_summarize.split()))
                    with col2:
                        st.metric(
                            "Summary Length", len(summary[0]["summary_text"].split())
                        )

                    # Download option
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"summary_{timestamp}.txt"
                    with open(filename, "w") as f:
                        f.write(f"ORIGINAL TEXT:\n\n{text_to_summarize}\n\n")
                        f.write(f"SUMMARY:\n\n{summary[0]['summary_text']}")

                    with open(filename, "r") as f:
                        st.download_button(
                            label="Download Summary",
                            data=f,
                            file_name=filename,
                            mime="text/plain",
                        )

                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        else:
            st.error(
                "Summarization model is not loaded. Please try reloading the page."
            )

    st.markdown("</div>", unsafe_allow_html=True)
