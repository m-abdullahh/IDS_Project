import cv2
import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np


with open("styles.css") as f:
    css = f.read()


def reshape_image(image):
    return image.reshape((-1, 3))


def kmeans_color_separation(image, k):
    reshaped_image = reshape_image(image)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_image)
    labels = kmeans.predict(reshaped_image)
    centroids = kmeans.cluster_centers_
    percentages = [(labels == i).sum() / len(labels) for i in range(k)]
    return labels, centroids, percentages


def generate_palette(image, k):
    labels, centroids, percentages = kmeans_color_separation(image, k)
    colors = [(int(c[0]), int(c[1]), int(c[2])) for c in centroids]

    palette = f" <style>{css}</style> <div  class='palette-container'>"

    for i, color in enumerate(colors):
        hex_code = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        palette += f"<div class='palette-item' style='background-color:{hex_code}'>{hex_code}</div>"

    palette += "</div>"
    return palette


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Colour Palette Generator</h1>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        file_uploader = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

        k = st.sidebar.number_input(
            "Number of Clusters:", min_value=2, max_value=10, value=5
        )

    if file_uploader:
        image = cv2.imdecode(
            np.frombuffer(file_uploader.read(), np.uint8), cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Downscale the image to a maximum dimension of 500 pixels
        max_dimension = 500
        image_height, image_width, _ = image.shape
        if image_height > max_dimension or image_width > max_dimension:
            scaling_factor = max_dimension / max(image_height, image_width)
            image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor)

        with st.sidebar:
            st.image(image, caption="Uploaded Image", width=300)

        palette = generate_palette(image, k)
        st.markdown(palette, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
