import streamlit as st
from llmware.models import ModelCatalog

# Title of the Streamlit app
st.title("Text Analysis Tool")

# Text input for the text to analyze
text_to_analyze = st.text_area("Enter the text to analyze:", "I am John Doe, I visited Peter, a therapist in the year 2019. I travelled to NYC for around 5 therapy sessions, I was advised to be more outgoing.")

# Text input for the queries
queries_input = st.text_area("Enter your queries (comma separated):", "consultant specialty, Consultant name, Consultant location, Consultant Advice")

# Convert the input queries to a list
queries_list = [query.strip() for query in queries_input.split(',')]

# Button to run the analysis
if st.button("Analyze"):
    # Load the model
    model = ModelCatalog().load_model("slim-extract-tool", sample=False, temperature=0.0, max_output=200)
    
    # Initialize the output dictionary
    output_dict = {}

    # Loop through the queries and call the model with the entire text for each query
    for j, query in enumerate(queries_list):
        st.write(f"Query {j+1}: {query}")
        response = model.function_call(text_to_analyze, function="extract", params=[query])
        output_dict.update(response["llm_response"])
        #if not response["llm_response"]:
           # st.write("No response")
       # st.write("Extract response: ", response["llm_response"])

    # Display the response on the screen
    st.write("Output Dictionary:")
    st.json(output_dict)
