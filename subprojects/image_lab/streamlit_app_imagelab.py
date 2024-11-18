import os
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison
from subprojects.image_lab.utils import refine_prompt, analyze_image, prompt2sound, prompt2video, enhance_image
from subprojects.image_lab.segmind_utils import FluxModel
from aux_utils.auto_instagram_publi import InstagramDescriptor  # Added import

# Directory Setup
INPUT_DIR = 'subprojects/image_lab/input_images'
OUTPUT_DIR = 'subprojects/image_lab/output_images'

# Create input and output directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    st.title("ImageLab 🎨")
    #cinematic_style = st.sidebar.toggle("Cinematic Style 🎥", True)
    generate_instagram = st.sidebar.toggle("Generate Instagram Description 📸", False)  # Added toggle

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    #setting toggle
    generate_video_prompt = st.sidebar.toggle("Generate Video Prompt 🎥", False)
    generate_audio_prompt = st.sidebar.toggle("Generate Sound Prompt 🎵", False)
    
    query = st.chat_input("Type a prompt for optimization...")
    
    if query:
        st.write("Refining the prompt...")
        #we write the prompt to the screen
        with st.chat_message("user"):
            st.write("Prompt:", query)
    
        # if cinematic_style:
        #     #open the cinematic_prompt.md file and read the contents to get the cinematic prompt
        #     with open("subprojects/image_lab/cinematic_prompt.md", "r") as f:
        #         cinematic_style_prompt = f.read()
                
        refined_prompt = refine_prompt(query) #, style_prompt=cinematic_style_prompt)
        st.session_state["refined_prompt"] = refined_prompt
        with st.chat_message("ai"):
            st.write("Refined prompt:", refined_prompt)
            #st.text_area(refined_prompt)
            
        # Create a video prompt
        if generate_video_prompt:
            with st.chat_message("ai"):
                st.write("Generating video prompt...")
                video_prompt = prompt2video(refined_prompt)
                st.write("Video prompt:", video_prompt)
            
        # Create a sound prompt
        if generate_audio_prompt:
            with st.chat_message("ai"):
                st.write("Generating sound prompt...")
                sound_prompt = prompt2sound(refined_prompt)
                st.write("Sound prompt:", sound_prompt)
        
        if generate_instagram:  # Handle Instagram description generation
            st.write("Generating Instagram description...")
            descriptor = InstagramDescriptor()
            instagram_description = descriptor.generate_description(refined_prompt)
            with st.chat_message("ai"):
                st.write("Instagram Description:", instagram_description)
                
        #print a button "Generate with Flux 1.1 Pro" that when clicked will generate the image
        
    
    if st.sidebar.button("Generate with Flux 1.1 Pro"):
            model=FluxModel()
            #we generate the image
            with st.spinner("Generating image..."):
                image = model.generate(prompt=st.session_state["refined_prompt"])
            st.toast("Image generated successfully!", icon="✅")
            #we save the image
            model.save(image)
            #we display the image
            st.image(image, caption='Generated Image.', use_column_width=True)
        
      
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        # Save the uploaded image to the input directory
        input_path = os.path.join(INPUT_DIR, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Buttons for upscaling and analyzing
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upscale Image"):
                st.write("Enhancing image...")
                enhanced_image_path = enhance_image(input_path)
                if enhanced_image_path and os.path.exists(enhanced_image_path):
                    original_image = Image.open(input_path)
                    enhanced_image = Image.open(enhanced_image_path)
                    st.write("### Compare Original and Enhanced Images")
                    image_comparison(
                        img1=original_image,
                        img2=enhanced_image,
                        label1="Original",
                        label2="Enhanced",
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True,
                    )
                    #save the image to the output directory
        
                else:
                    st.error("Failed to enhance the image. Please try again.")
        
        with col2:
            if st.button("Analyze Image"):
                st.write("Analyzing image...")
                ingredients = analyze_image(uploaded_file)
                st.write("Ingredients found:", ingredients)

if __name__ == "__main__":
    main()