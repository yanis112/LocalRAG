import streamlit as st
from subprojects.fridge2plate.utils import analyze_image, generate_recipe

def main():
    st.title("Fridge to Recipe App")
    st.write("Upload an image of your fridge and get a recipe idea!")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Analyzing image...")

        # Analyze the image to get ingredients
        ingredients = analyze_image(uploaded_file)
        st.write("Ingredients found:", ingredients)

        # Generate a recipe based on the ingredients
        st.write("Generating recipe...")
        recipe = generate_recipe(ingredients)
        st.markdown(recipe)

if __name__ == "__main__":
    main()