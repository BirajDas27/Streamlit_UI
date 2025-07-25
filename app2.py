import streamlit as st
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Make sure to import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


# Define paths
base_dir = 'C:\\Users\\91757\\split_data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Function to get categories and image counts
def get_category_info(directory):
    if os.path.exists(directory):
        categories = os.listdir(directory)
        info = {}
        for category in categories:
            category_path = os.path.join(directory, category)
            if os.path.isdir(category_path):
                info[category] = len(os.listdir(category_path))
        return info
    else:
        return None

# Sidebar menu options
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Project Description", "Model Performance", "Model Prediction", "Used Augmentation"],
    )

# Content for "Group A"
if selected == "Model Performance":
    model_selected = st.selectbox("Select a Model", ["MobileNet-V2", "DenseNet", "Inception"])
    st.markdown(
        f"""
        <h1 style="font-size: 45px; text-decoration: underline; text-decoration-color: white; thickness: 3px">
            You have selected {model_selected}
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    #mobilenet V2
    if model_selected == "MobileNet-V2":
        # Display dataset structure
        st.markdown(
            """
            <h2 style="font-size: 40px; text-decoration: underline; text-decoration-style: dotted">
                Dataset Structure
            </h2>
            """,
            unsafe_allow_html=True
        )
        dataset_structure = {
            "Train": get_category_info(train_dir),
            "Validation": get_category_info(validation_dir),
            "Test": get_category_info(test_dir),
        }

        for split, info in dataset_structure.items():
            if info:
                st.subheader(f"{split} Split")
                for category, count in info.items():
                    st.write(f"- **{category}**: {count} images")
            else:
                st.write(f"**{split} Split**: Directory not found!")


        # Results achieved
        st.markdown(
            """
            <h2 style="font-size: 40px; text-decoration: underline; text-decoration-style: dotted">
                Results
            </h2>
            """,
            unsafe_allow_html=True
        )
        #accuracy
        st.subheader("MobileNet-V2 Accuracy (per epoch)")
        image = Image.open("mobileNet-acc.png")
        st.image(image, caption="Model accuracy per epoch chart", use_container_width=True)

        #confusion matrix
        st.subheader("MobileNet-V2 Confusion matrix")
        image = Image.open("mobileNet-confusion.png")
        st.image(image, caption="Model confusion matrix", use_container_width=True)

        st.success(f"Precision: {0.91}")
        st.info(f"Recall: {0.89}")
        st.warning(f"F1-score: {0.90}")



    #denseNet    
    if model_selected == "DenseNet":
        # Display dataset structure
        st.markdown(
            """
            <h2 style="font-size: 40px; text-decoration: underline; text-decoration-style: dotted">
                Dataset Structure
            </h2>
            """,
            unsafe_allow_html=True
        )
        dataset_structure = {
            "Train": get_category_info(train_dir),
            "Validation": get_category_info(validation_dir),
            "Test": get_category_info(test_dir),
        }

        for split, info in dataset_structure.items():
            if info:
                st.subheader(f"{split} Split")
                for category, count in info.items():
                    st.write(f"- **{category}**: {count} images")
            else:
                st.write(f"**{split} Split**: Directory not found!")

        # Results achieved
        st.markdown(
            """
            <h2 style="font-size: 40px; text-decoration: underline; text-decoration-style: dotted">
                Results
            </h2>
            """,
            unsafe_allow_html=True
        )

        #accuracy
        st.subheader("DenseNet Accuracy (per epoch)")
        image = Image.open("denseNet-acc.png")
        st.image(image, caption="Model accuracy per epoch chart", use_container_width=True)

        #confusion matrix
        st.subheader("DenseNet Confusion matrix")
        image = Image.open("denseNet-confusion.png")
        st.image(image, caption="Model confusion matrix", use_container_width=True)

        st.success(f"Precision: {0.94}")
        st.info(f"Recall: {0.93}")
        st.warning(f"F1-score: {0.93}")

        


    #inception
    if model_selected == "Inception":
        # Display dataset structure
        st.markdown(
            """
            <h2 style="font-size: 40px; text-decoration: underline; text-decoration-style: dotted">
                Dataset Structure
            </h2>
            """,
            unsafe_allow_html=True
        )
        dataset_structure = {
            "Train": get_category_info(train_dir),
            "Validation": get_category_info(validation_dir),
            "Test": get_category_info(test_dir),
        }

        for split, info in dataset_structure.items():
            if info:
                st.subheader(f"{split} Split")
                for category, count in info.items():
                    st.write(f"- **{category}**: {count} images")
            else:
                st.write(f"**{split} Split**: Directory not found!")

        # Results achieved
        st.markdown(
            """
            <h2 style="font-size: 40px; text-decoration: underline; text-decoration-style: dotted">
                Results
            </h2>
            """,
            unsafe_allow_html=True
        )
        st.subheader("Inception Accuracy (per epoch)")
        image = Image.open("inception-acc.png")
        st.image(image, caption="Model accuracy per epoch chart", use_container_width=True)

        st.subheader("Inception Confusion matrix")
        image = Image.open("inception-confusion.png")
        st.image(image, caption="Model confusion matrix", use_container_width=True)

        st.success(f"Precision: {0.88}")
        st.info(f"Recall: {0.85}")
        st.warning(f"F1-score: {0.86}")


# Content for "Group B"
if selected == "Model Prediction":
    model_selected = st.selectbox("Select a Model", ["MobileNet-V2", "DenseNet", "Inception"])
    st.markdown(
        f"""
        <h1 style="font-size: 45px; text-decoration: underline; text-decoration-color: white; thickness: 3px">
            You have selected {model_selected}
        </h1>
        """,
        unsafe_allow_html=True
    )

    if model_selected == "MobileNet-V2":
        # Loading trained model
        @st.cache_resource
        def load_trained_model():
            model_path = "breast_cancer_mobilenetv2.keras"
            return load_model(model_path)

        model = load_trained_model()


        if model is not None:
            # Image input
            uploaded_image = st.file_uploader("Upload an Image for prediction", type=["jpg", "png", "jpeg"])

            if uploaded_image is not None:
                # Display the uploaded image
                img = Image.open(uploaded_image)
                img = img.resize((224, 224))  # Resize image for model input
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array /= 255.0  # Normalize the image

                # Make a prediction
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Preprocess the image for the model
                image = image.resize((224, 224))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Make a prediction
                prediction = model.predict(image_array)
                predicted_class = "Malignant" if prediction[0] > 0.5 else "Benign"

                st.info(f"Prediction: {predicted_class}")
                st.success(f"Confidence: {prediction[0][0]:.2f}")

        else:
            st.write(f"No saved model found for {model_selected}.")

    if model_selected == "DenseNet":
        # Loading trained model
        @st.cache_resource
        def load_trained_model():
            model_path = "breast_cancer_densenet.h5"
            return load_model(model_path)

        model = load_trained_model()

        if model is not None:
            # Image input
            uploaded_image = st.file_uploader("Upload an Image for prediction", type=["jpg", "png", "jpeg"])

            if uploaded_image is not None:
                # Display the uploaded image
                img = Image.open(uploaded_image)
                img = img.resize((224, 224))  # Resize image for model input
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array /= 255.0  # Normalize the image

                # Make a prediction
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Preprocess the image for the model
                image = image.resize((224, 224))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Make a prediction
                prediction = model.predict(image_array)
                predicted_class = "Malignant" if prediction[0] > 0.5 else "Benign"

                st.info(f"Prediction: {predicted_class}")
                st.success(f"Confidence: {prediction[0][0]:.2f}")

        else:
            st.write(f"No saved model found for {model_selected}.")

    if model_selected == "Inception":
        # Loading trained model
        @st.cache_resource
        def load_trained_model():
            model_path = "breast_cancer_inception.keras"
            return load_model(model_path)

        model = load_trained_model()

        if model is not None:
            # Image input
            uploaded_image = st.file_uploader("Upload an Image for prediction", type=["jpg", "png", "jpeg"])

            if uploaded_image is not None:
                # Display the uploaded image
                img = Image.open(uploaded_image)
                img = img.resize((224, 224))  # Resize image for model input
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array /= 255.0  # Normalize the image

                # Make a prediction
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Preprocess the image for the model
                image = image.resize((224, 224))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Make a prediction
                prediction = model.predict(image_array)
                predicted_class = "Malignant" if prediction[0] > 0.5 else "Benign"

                st.info(f"Prediction: {predicted_class}")
                st.success(f"Confidence: {prediction[0][0]:.2f}")

        else:
            st.write(f"No saved model found for {model_selected}.")


#content for group C
if selected == "Used Augmentation":

    def resize_image(image_path, size=(300, 250)):
        try:
            # Open the image using Pillow
            image = Image.open(image_path)
            # Resize the image to the specified size
            return image.resize(size)
        except FileNotFoundError:
            st.error(f"Error: The file {image_path} was not found.")
        except Exception as e:
            st.error(f"An error occurred while processing the image {image_path}: {e}")
            return None

    st.markdown(
        f"""
        <h1 style="font-size: 45px; text-decoration: underline; text-decoration-color: white; thickness: 3px">
            Description of Augmentation
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <h1 style="font-size: 35px; text-decoration: underline; text-decoration-style: dotted; text-decoration-color: white; thickness: 3px">
            Augmentation Techniques Used
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.write("""
    In this project, we utilized the following augmentation techniques to enhance the dataset while preserving the integrity of the microscopic images:
    """)
    
    #rotation
    st.subheader("1. Rotation")
    st.write("""
    Rotation involves rotating the image by a certain angle. This helps the model learn from different orientations of the same image, increasing its robustness to variations in image alignment. 
    For example, a 90-degree rotation would allow the model to classify images irrespective of their initial orientation.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og1.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/rotated.jpeg"), caption="Augmented Image (Rotated)", use_container_width=True)
    
    #H-flip
    st.subheader("2. Horizontal Flip")
    st.write("""
    Horizontal flipping mirrors the image along the vertical axis. This augmentation is particularly useful when the dataset lacks symmetry in orientations, providing the model with a broader range of patterns to learn from.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og2.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/HFlip.jpeg"), caption="Augmented Image (Horizontal)", use_container_width=True)
    
    #V-flip
    st.subheader("3. Vertical Flip")
    st.write("""
    Vertical flipping mirrors the image along the horizontal axis. Similar to horizontal flipping, this technique introduces variations in orientation without altering the structural features of the cells.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og2.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/VFlip.jpeg"), caption="Augmented Image (Vertical)", use_container_width=True)

    st.markdown(
        f"""
        <h1 style="font-size: 35px; text-decoration: underline; text-decoration-style: dotted; text-decoration-color: white; thickness: 3px">
            Augmentation Techniques Not Used
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.write("""
    While many augmentation techniques exist, some were intentionally avoided in this project due to their potential to alter or distort the nature of the data, particularly for microscopic images where accuracy is critical. 
    The following techniques were not used:
    """)
    
    #grayscale
    st.subheader("1. Grayscaling")
    st.write("""
    Grayscaling converts an image to grayscale, removing color information. Since microscopic images often rely on subtle color differences to distinguish features, grayscaling could remove valuable data critical for classification.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og3.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/gray.jpeg"), caption="Augmented Image (Grayscaled)", use_container_width=True)
    
    #crop
    st.subheader("2. Cropping")
    st.write("""
    Cropping alter the spatial dimensions of the image. For microscopic images, cropping might cut out important parts of the cells, leading to loss of critical details necessary for classification.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og2.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/crop.jpeg"), caption="Augmented Image (cropped)", use_container_width=True)
    
    #filter
    st.subheader("3. Applying Filters")
    st.write("""
    Applying filters (e.g., blurring or sharpening) might introduce artificial patterns that are not representative of real-world data. This could lead to overfitting or misclassification as the model might learn features that do not exist in actual microscopic images.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og1.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/filter.jpg"), caption="Augmented Image (Filtered)", use_container_width=True)
    
    #shear
    st.subheader("4. Shearing")
    st.write("""
    Shearing distorts the shape of the image along one axis, which could alter the morphology of the cells in the microscopic images. Since cell morphology is crucial for classification, this technique was avoided to preserve the original structure of the data.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og1.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/shear.jpg"), caption="Augmented Image (Sheared)", use_container_width=True)
    
    #zoom
    st.subheader("5. Zooming Out")
    st.write("""
    Zooming out alters spatial dimensions between cells, potentially changing an important feature for detection. For microscopic images, maintaining the overall structure and relationships within the image is important, making zooming out unsuitable.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_image("eg/og1.jpeg"), caption="Original Image", use_container_width=True)
    with col2:
        st.image(resize_image("eg/zoomOut.jpg"), caption="Augmented Image (Zoomed out)", use_container_width=True)
    
    st.write("""
    By carefully selecting augmentation techniques, we ensured that the dataset was enhanced in ways that preserved the essential features of the images and improved model performance without introducing unnecessary distortions.
    """)

    unsafe_allow_html = True



#content for group D
if selected == "Project Description":
    st.markdown(
        f"""
        <h1 style="font-size: 45px; text-decoration: underline; text-decoration-color: white; thickness: 3px">
            Breast Cancer Cell Detection Using Deep Learning and Image Augmentation
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    st.header("About the Project")
    st.write("""
    The goal of this project is to detect breast cancer cells by classifying them into benign and malignant categories. 
    To enhance the quality of the base dataset, several augmentation techniques were applied, including rotation, horizontal flip, and vertical flip. 
    These techniques were chosen to artificially increase the dataset size while maintaining the integrity of the images.
    """)
    
    st.write("""
    The dataset was obtained from **Scikit-Hub**, and it was split into training, testing, and validation sets with a distribution of 75%, 15%, and 15%, respectively. 
    This ensured that the models were trained on a sufficiently large dataset while having enough data for testing and validating the model performance.
    """)
    
    st.write("""
    Three different models—MobileNetV2, DenseNet, and Inception—were evaluated in this project to determine the best performing model for breast cancer cell detection. 
    Each model was tested on the augmented dataset to assess their accuracy in classifying the images. 
    The results from these models provide valuable insights into their effectiveness in detecting breast cancer cells in medical imagery.
    """)
    
    st.write("""
    **Why Limited Augmentation Techniques Were Used:**
    In this project, we focused on augmentation techniques like rotation, horizontal flip, and vertical flip to avoid introducing unwanted distortions to the dataset. 
    Techniques like grayscaling, cropping, zooming out, applying filters, and shearing were avoided because they could potentially change the nature of the data in ways that could interfere with accurate classification. 
    For instance, cropping and zooming out could alter the spatial features of the cell images, while grayscaling or applying filters might remove important color information, especially in images where subtle color differences are crucial for classification. 
    Additionally, shearing could distort the cell structures, making it harder for the models to learn meaningful patterns. 
    By sticking to these simpler augmentations, we ensured the integrity of the microscopic images and preserved their essential features.
    """)