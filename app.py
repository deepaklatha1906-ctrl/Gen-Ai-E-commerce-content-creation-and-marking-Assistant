import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from utils_llm import generate_text, generate_chat_response
from utils_cv import analyze_image
from utils_image_gen import generate_marketing_image
from utils_recommender import get_recommendations

load_dotenv()
st.set_page_config(page_title="E-Commerce AI Assistant", layout="wide")

st.title("🛒 Generative AI E-Commerce Marketing Assistant")
# ✅ UPDATED: Reflects current API dependencies

tabs = st.tabs(["📝 Content Creation", "📊 Review Analysis", "🎯 Recommendations", "🎨 Ad Image Gen", "💬 AI Chatbot"])

# --- Tab 1: Content Creation ---
with tabs[0]:
    st.header("Product Content Creation")
    col1, col2 = st.columns([1, 1])
    with col1:
        product_name = st.text_input("Product Name")
        category = st.text_input("Category")
        features = st.text_area("Key Features (comma separated)")
        uploaded_file = st.file_uploader("Upload Product Image (Optional)", type=["jpg", "png", "jpeg"])
        
        image_description = ""
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image")
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing image using Gemini Vision..."):
                    try:
                        image_description = analyze_image(uploaded_file.getvalue())
                        st.success("Image analyzed!")
                        st.write(f"**Detected:** {image_description}")
                    except Exception as e:
                        st.error(str(e))

    with col2:
        if st.button("✨ Generate Content"):
            if not product_name:
                st.warning("Please enter at least a product name.")
            else:
                with st.spinner("Generating SEO-optimized content using Gemini..."):
                    prompt = f"""
                    Product Name: {product_name}
                    Category: {category}
                    Features: {features}
                    Visual Description from Image: {image_description}
                    
                    Please generate:
                    1. SEO-friendly Product Title
                    2. Detailed Product Description (3 paragraphs)
                    3. Bulleted Feature List
                    4. 10 SEO Keywords and Tags
                    5. 3 Social Media Captions with Hashtags
                    6. Short Advertisement Copy
                    """
                    response = generate_text(prompt)
                    st.markdown(response)

# --- Tab 2: Review Analysis ---
with tabs[1]:
    st.header("Customer Review Analysis")
    reviews = st.text_area("Paste Customer Reviews (one per line)", height=200)
    if st.button("📊 Analyze Reviews"):
        if not reviews:
            st.warning("Please paste some reviews.")
        else:
            with st.spinner("Analyzing sentiment and summarizing..."):
                prompt = f"""
                Analyze the following customer reviews:
                {reviews}
                Provide:
                1. Overall Sentiment (Positive, Negative, or Mixed)
                2. Key Strengths mentioned by customers
                3. Areas for Improvement / Common Complaints
                4. A concise 2-sentence summary of the feedback
                """
                st.markdown(generate_text(prompt))

# --- Tab 3: Recommendations ---
with tabs[2]:
    st.header("Cross-Selling Recommendations")
    st.write("Enter a product description or name to find related products for cross-selling.")
    query = st.text_input("Product Query")
    if st.button("🎯 Get Recommendations"):
        if not query:
            st.warning("Please enter a product query.")
        else:
            with st.spinner("Searching vector database..."):
                recs = get_recommendations(query, top_k=2)
                if recs:
                    st.success("Recommended Products:")
                    for rec in recs:
                        st.markdown(f"**{rec['name']}** ({rec['category']})\n- {rec['description']}")
                else:
                    st.info("No highly relevant recommendations found.")

# --- Tab 4: Image Generation ---
with tabs[3]:
    st.header("Promotional Marketing Creative Generation")
    img_prompt = st.text_area("Describe the marketing image you want to generate", 
                              placeholder="e.g., A sleek wireless headphone on a minimalist wooden desk, soft studio lighting, 4k resolution, photorealistic, advertising style")
    if st.button("🎨 Generate Marketing Image"):
        if not img_prompt:
            st.warning("Please provide an image description.")
        else:
            with st.spinner("Generating image using Hugging Face Stable Diffusion... (May take 15-30s)"):
                try:
                    enhanced_prompt = f"Professional e-commerce product photography, {img_prompt}, high quality, highly detailed, 8k resolution, cinematic lighting, trending on artstation"
                    generated_image = generate_marketing_image(enhanced_prompt)
                    st.image(generated_image, caption="Generated Marketing Creative")
                    st.success("Image generated successfully!")
                except Exception as e:
                    st.error(str(e))

# --- Tab 5: AI Chatbot ---
with tabs[4]:
    st.header("💬 AI Marketing Chatbot Assistant")
    st.write("Ask anything about your product marketing, SEO, or content strategy!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are an expert e-commerce marketing assistant. Help the user with SEO, content creation, and marketing strategies."}]
        
    for msg in st.session_state.chat_history:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Keep last 6 messages for context window management
                context_messages = st.session_state.chat_history[-6:]
                response = generate_chat_response(context_messages)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})