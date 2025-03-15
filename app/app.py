#________________________________________________________________________________________________
#                                IMPORTING DEPENDENCIES AND API KEYS
#________________________________________________________________________________________________

import pandas as pd
import re
import string  
import nltk  
import torch
import openai
import streamlit as st
import contractions  
import os

from collections import Counter
from nltk.tokenize.toktok import ToktokTokenizer  # Toktok tokenizer for tokenization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit.components.v1 as components

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize Toktok tokenizer
tokenizer = ToktokTokenizer()
pd.set_option('display.max_columns', None)

#storing my api key to use for the project
api_key = st.secrets["api_keys"]["my_api_key"]
openai.api_key = api_key

#________________________________________________________________________________________________
#                              INITIATE STREAMLIT APP AND SET THEME
#________________________________________________________________________________________________

# Set the page configuration without the theme argument
st.set_page_config(
    page_title="EduLens",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_path = os.path.join(os.path.dirname(__file__), "assets/logo.png")
# Add custom CSS to style the logo
st.markdown("""
    <style>
        .logo {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 100px; /* Adjust size as needed */
        }
    </style>
""", unsafe_allow_html=True)

# Display the logo in the top left corner
st.image(logo_path, use_container_width=False, width=100, caption=None)

# Custom CSS for styling the app elements
st.markdown(
    """
    <style>
    /* Customize the main background and text */
    .css-1d391kg {
        background-color: #f0f0f0; /* Background color for the main body */
        color: #333333; /* Text color */
    }
    /* Style the header */
    .stTitle {
        color: #1e90ff; /* Header color */
    }
    /* Customize buttons */
    .stButton>button {
        background-color: #1e90ff; /* Button background color */
        color: white; /* Button text color */
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #007bb5; /* Hover effect for buttons */
    }
    /* Style the sidebar */
    .stSidebar {
        background-color: #e0e0e0; /* Sidebar background */
    }
    /* Style the download button */
    .stDownloadButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stDownloadButton>button:hover {
        background-color: #007bb5;
    }
    </style>
    """,
    unsafe_allow_html=True)

# Streamlit App
st.title("EduLens: Student Feedback Analyzer") 

st.write('''This app is designed to help you explore and analyze student feedback efficiently and
          meaningfully. With EduLens, you can clean and preprocess feedback, perform sentiment analysis 
         to distinguish between satisfactory and dissatisfactory reviews, and group similar feedback into
          clusters using advanced text analytics techniques. You can customize the analysis by excluding specific 
         words, gain insights into frequently used terms, and even review feedback grouped by sentiment and cluster 
         labels. Once your analysis is complete, you can easily download the enriched dataset, complete with sentiment
          and cluster information, for further use.''')

st.write("To get started, simply upload your dataset, follow the guided steps to clean and analyze your data, and unlock actionable insights into student feedback!")

#________________________________________________________________________________________________
#                              INSTALL GOOGLE ANALYTICS ON THE SITE
#________________________________________________________________________________________________

# Google Analytics Script
ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-C7QZF0QS0R"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-C7QZF0QS0R');
</script>
"""

# Inject script into the Streamlit app
st.components.html(ga_script, height=0)


#________________________________________________________________________________________________
#                     INITIATE FUNCTIONS AND VARIABLES THAT WILL BE USED LATER
#________________________________________________________________________________________________

#Bring in my text cleaning funtion that i have made for the previous assignments
def clean_text_clustering(df, text_columns, custom_stopwords = None):
    # Convert text columns to lowercase and strip whitespace
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                # Apply cleaning steps only to string values
                df[col] = df[col].astype(str).str.lower().str.strip()

                # Expand contractions
                df[col] = df[col].apply(lambda x: contractions.fix(x) if isinstance(x, str) else x)

                # Remove punctuation
                df[col] = df[col].str.translate(str.maketrans('', '', string.punctuation))

                # Remove anything that is not a word
                df[col] = df[col].apply(lambda text: re.sub(r'[^a-zA-Z ]+', '', text) if isinstance(text, str) else text)

                df[col] = df[col].astype(str).apply(
                    lambda x: ' '.join(word for word in word_tokenize(x.lower()) if word not in custom_stopwords)
                 )

    return df

#Bring in my text cleaning funtion that i have made for the previous assignments
def clean_text_sentiment(df, text_columns):
    # Convert text columns to lowercase and strip whitespace
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                # Apply cleaning steps only to string values and convert to lower / strip whitespace
                df[col] = df[col].astype(str).str.lower().str.strip()

                # Expand contractions
                df[col] = df[col].apply(lambda x: contractions.fix(x) if isinstance(x, str) else x)

                # Remove non-essential special characters
                df[col] = df[col].apply(lambda text: re.sub(r'[^\w\s]', '', text) if isinstance(text, str) else text)

    return df

# Function to calculate word frequency
def word_frequency(text, N=15):
    tokens = word_tokenize(text.lower())  # Tokenizing and lowercasing text
    frequency = Counter(tokens)  # Calculating the frequency of each word
    return frequency.most_common(N)  # Returning the top N most frequent words

# Function to predict sentiment label
def get_sentiment_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    sentiment = torch.argmax(logits, dim=-1).item()
    return "Satisfactory" if sentiment == 1 else "Disatisfactory"

#our elbow locator for k means clustering
def elbow_locator (text_vec_matrix):
    # create a kmeans initialization dictionary
    kmeans_kwargs = {
        "init": "random",
        "n_init": 30,
        "max_iter": 500,
        "random_state": 42,
    }

    # create empty list for SSE values
    sse = []

    # create loop to fit kmeans clustering analysis of k of size 1-11, add SSE values for each model to the list
    for k in range(1,7):
        kmeans = KMeans(n_clusters= k, **kmeans_kwargs)
        kmeans.fit(text_vec_matrix)
        sse.append(kmeans.inertia_)

    #create our knee locator object and show it
    kl = KneeLocator(range(1, 7), sse, curve='convex', direction='decreasing')
    return kl.elbow

#get our cluster labels
def get_cluster_labels(df_cleaned, df_original, text_vec_matrix, n_clusters, vectorize):
    # Perform KMeans clustering
    km = KMeans(
        n_clusters=n_clusters,
        max_iter=500,
        n_init=30,
        init='random',
        random_state=42
    ).fit(text_vec_matrix)

    # Assign cluster labels to the cleaned DataFrame
    df_cleaned['kmeans_cluster'] = km.labels_

    # Create a DataFrame to combine the cleaned and original data with clusters
    df_combined = df_cleaned[['kmeans_cluster']].copy()
    df_combined['original_review'] = df_original['review']  # Original text
    df_combined['sentiment'] = df_cleaned['sentiment'] 

    # Select top reviews for each cluster
    review_clusters = (
        df_combined
        .sort_values(by=['kmeans_cluster', 'sentiment'], ascending=False)
        .groupby('kmeans_cluster')
        .head(10)
    )

    # Extract top features for each cluster
    feature_names = vectorize.get_feature_names_out()
    topn_features = 15
    ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for cluster_num in range(n_clusters):
        # Key features for the cluster
        key_features = [
            feature_names[index]
            for index in ordered_centroids[cluster_num, :topn_features]
            if 0 <= index < len(feature_names)
        ]

        # Top reviews (original, not cleaned)
        reviews = review_clusters[review_clusters['kmeans_cluster'] == cluster_num]['original_review'].values.tolist()

        cluster_name = chatGPT_cluster_labels(key_features, reviews)
        df_cleaned['cluster and feedback'] = cluster_name

        print(cluster_name)
        # Display cluster header
        st.markdown(f"#### CLUSTER #{cluster_num + 1}: {cluster_name}")
        
        # Display key features
        st.markdown("**Top Contributing Features:**")
        st.write(", ".join(key_features))
        
        # Display top reviews in an expander for readability
        with st.expander("Top Feedback"):
            for idx, review in enumerate(reviews):
                st.markdown(f"{idx + 1}. {review}")
        # Add a horizontal separator
        st.markdown("---")
    
    return df_cleaned

#Create a open-ai API request to get cluster labels and feedback/ a summary for each cluster
def chatGPT_cluster_labels(features, feedback):

    # Create a chat completion request
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the model
        messages=[
            {"role": "user", "content": f'''Given a list of top features and feedback from students create a cluster name.
            followed by a new line with your own feedback about what is going well or what could be improved. Pay special attention to the 
             feedback specifically. \nFeatures: \n{features}\n Feedback: \n {feedback}'''}
        ]
    )

    #store chat gpts response
    # response_content = cluster_name.choices[0].message['content'].strip()

    return response.choices[0].message.content

# Initialize NLTK stopwords
default_stopwords = set(stopwords.words("english"))

#create a custom stopwords list
extra_stopwords = set([ 'course','class', 'professor', 'professors', 'prof', 'profs', 'philosophyreligion', 'get',
                    'teacher', 'students', 'week', 'courses', 'also', 'equations', 'machine',  
                    'first', 'one', 'classes', 'us', 'go', 'spanish', 'physics', 'data', 'science',
                    'school', 'high', 'coursera', 'astronomy','andrew', 'ng', 'algorithms', 'programming',
                     'lecture', 'lectures', 'duke', 'university','eating' ,'sooo', 'cute', 'math', 
                     ])

custom_stopwords = default_stopwords.union(extra_stopwords)
#________________________________________________________________________________________________
#                             1. LOAD IN THE FILE AND PERFORM CLEANING
#________________________________________________________________________________________________

# I wanted to create a restart button that clears all output for a new 
if st.button("Restart App"):
    st.session_state.clear()  # Clear all session state data
    st.rerun()  # Restart the app to reset the UI

# Display a prompt message asking the user to upload a file with feedback
st.markdown("""
    Please upload a file containing **feedback only**. 
    The file should be in CSV, TXT, or Excel format.
    """)

# File uploader
uploaded_file = st.file_uploader("Upload your file:", type=["csv", "txt", "xlsx"])

# Initialize session state for stopwords and DataFrame
if "custom_stopwords" not in st.session_state:
    st.session_state["custom_stopwords"] = custom_stopwords
if "df_original" not in st.session_state:
    st.session_state["df_original"] = None
if "df_cleaned_clustering" not in st.session_state:
    st.session_state["df_cleaned_clustering"] = None
if "df_cleaned_sentiment" not in st.session_state:
    st.session_state["df_cleaned_sentiment"] = None
if "df_labled" not in st.session_state:
    st.session_state["df_labled"] = None

# Check the uploaded file and see what the format is and then apply cleaning
if uploaded_file:
    # Only process the file if it's not already loaded
    if st.session_state["df_original"] is None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            df = pd.DataFrame({"review": content.splitlines()})  # Treat each line as a review
        
        # Ensure the review column exists
        if "review" not in df.columns:
            df.rename(columns={df.columns[0]: "review"}, inplace=True)

        st.session_state["df_original"] = df  # Save the original DataFrame

        #create our cleaned dataframe that has been cut down for clustering
        st.session_state["df_cleaned_clustering"] = clean_text_clustering(
            st.session_state["df_original"].copy(), text_columns = ["review"], custom_stopwords = st.session_state["custom_stopwords"])

        #create our cleaned dataframe for our sentiment analysis that has basic cleaning done
        st.session_state["df_cleaned_sentiment"] = clean_text_sentiment(st.session_state["df_original"].copy(), text_columns = ["review"])

    # Always display the original data 
    st.subheader("Original Feedback Data")
    st.write(st.session_state["df_original"])

#________________________________________________________________________________________________
#                       2.  APPLY SENTIMENT ANALYSIS FROM HUGGING FACE
#________________________________________________________________________________________________

#Load in the hugging face mdoel we want to use for our sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Add a toggle to control sentiment analysis
if "run_sentiment" not in st.session_state:
    st.session_state["run_sentiment"] = False

if st.button("Run Sentiment Analysis"):
    # Enable sentiment analysis
    st.write("Running our sentiment analysis, this may take a minute...")
    st.session_state["run_sentiment"] = True

if st.session_state["run_sentiment"]:
    # Ensure the cleaned DataFrame exists
    if st.session_state.get("df_cleaned_sentiment") is not None:
        # Create a new DataFrame for labeled data without altering the original
        st.session_state["df_labeled"] = st.session_state["df_original"].copy()
        
        # Use the cleaned data for sentiment analysis but attach the sentiment column to the labeled DataFrame
        st.session_state["df_labeled"]["sentiment"] = st.session_state["df_cleaned_sentiment"]['review'].apply(get_sentiment_label)

        # Add our sentiment labels to the cleaned dataframe as well
        st.session_state["df_cleaned_clustering"]["sentiment"] = st.session_state["df_labeled"]["sentiment"]
        # Display the labeled DataFrame with sentiment
        st.subheader("Sentiment Analysis Results")
        st.write(st.session_state["df_labeled"])

        # Display sentiment percentages
        st.subheader("Overall Sentiment Percentages")
        sentiment_counts_percentage = st.session_state["df_labeled"]['sentiment'].value_counts(normalize=True) * 100
        sentiment_counts_percentage = sentiment_counts_percentage.map("{:.2f}%".format)
        st.write(sentiment_counts_percentage)
    else:
        st.warning("No cleaned data available. Please upload and process your data first.")
else:
    st.info("Click 'Run Sentiment Analysis' to get labels that are 'Disatisfactory' or 'Satisfactory'.")


#________________________________________________________________________________________________
#                       3. INITIATE TOP 15 WORDS AND CHECK TO REMOVE MORE
#________________________________________________________________________________________________

# Add a toggle to control sentiment analysis
if "top_features" not in st.session_state:
    st.session_state["top_features"] = False

if st.button("Show Top Word Features"):
    # Display explanations as before
    st.subheader("Analyzing Top Features for Clustering")
    st.write("Understanding the most common words helps identify terms that could impact clustering.")
    st.write("Removing high-frequency, non-distinctive words improves cluster quality.")
    st.session_state["top_features"] = True

if st.session_state.get("df_cleaned_clustering") is not None and st.session_state["top_features"]:
    # Display initial top 15 words
    all_text = ' '.join(st.session_state["df_cleaned_clustering"]['review'].dropna().astype(str))
    top_words = word_frequency(all_text)

    st.subheader("Top 15 Most Common Words (Excluding Current Stopwords)")
    for word, count in top_words:
        st.write(f"- **{word}**: {count} occurrences")

    # Allow user to specify additional stopwords
    exclude_words = st.text_input("Enter words to exclude (separate by commas):", value="")

    if st.button("Exclude Words"):
        if exclude_words:
            # Add new custom stopwords
            new_stopwords = {word.strip().lower() for word in exclude_words.split(",")}
            if "custom_stopwords" in st.session_state:
                st.session_state["custom_stopwords"] = st.session_state["custom_stopwords"].union(new_stopwords)
            else:
                st.session_state["custom_stopwords"] = new_stopwords

            # Re-clean the DataFrame
            st.session_state["df_cleaned_clustering"] = clean_text_clustering(
                st.session_state["df_original"].copy(), ["review"], st.session_state["custom_stopwords"]
            )

            st.success("Stopwords updated and removed!")

            # Display updated analysis
            all_text = ' '.join(st.session_state["df_cleaned_clustering"]['review'].dropna().astype(str))
            top_words = word_frequency(all_text)

            st.subheader("Updated Top 15 Most Common Words (After Exclusions)")
            for word, count in top_words:
                st.write(f"- **{word}**: {count} occurrences")
        else:
            st.warning("No words provided for exclusion, or be sure to separate them by commas.")

else:
    st.info("Click 'Top Word Features' to show the top 15 words and exclusion options for clustering analysis.")



#________________________________________________________________________________________________
#       4. K-MEANS CLUSTERING, FEATURE ENGINEERING, AND OPEN-AI FEEDBACK FOR EACH CLUSTER
#________________________________________________________________________________________________

if "run_clustering" not in st.session_state:
    st.session_state["run_clustering"] = False

if st.button("Run Clustering Analysis"):
    st.write("Running our clustering analysis, this may take a minute...")
    st.session_state["run_clustering"] = True

    #Begin our feature engineering
    try:
        # Vectorize the satisfactory and dissatisfactory reviews
        st.session_state["sat_vectorize"] = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        st.session_state["disat_vectorize"] = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

        # Filter reviews based on sentiment
        st.session_state["df_satisfactory"] = st.session_state["df_cleaned_clustering"][st.session_state["df_cleaned_clustering"]["sentiment"] == 'Satisfactory']
        st.session_state["df_disatisfactory"] = st.session_state["df_cleaned_clustering"][st.session_state["df_cleaned_clustering"]["sentiment"] == 'Disatisfactory']

        # Vectorize each category
        st.session_state["sat_matrix"] = st.session_state["sat_vectorize"].fit_transform(st.session_state["df_satisfactory"]['review'])
        st.session_state["disat_matrix"] = st.session_state["disat_vectorize"].fit_transform(st.session_state["df_disatisfactory"]['review'])

    except Exception as e:
        # Display the error only if an exception is raised
        st.error(f"An error occurred during feature engineering: {e}")
    

#Begin the clustering analsis
if st.session_state["run_clustering"]:
    # Run clustering for satisfactory comments
    if len(st.session_state["df_satisfactory"]) < 10:
        st.write("There was not enough satisfactory feedback to cluster, here are the satisfactory comments: ")

        # Display top reviews in an expander for readability
        with st.expander("Top Feedback"):
            for idx, review in enumerate(st.session_state['df_labeled']['review'].loc[st.session_state['df_satisfactory'].index]):
                st.markdown(f"{idx + 1}. {review}")
        # Add a horizontal separator
        st.markdown("---")

        # Create a DataFrame for sat_clusters with NA as the cluster name
        sat_clusters = st.session_state['df_labeled']['review'].loc[st.session_state['df_satisfactory'].index].to_frame(name='review')
        sat_clusters['cluster and feedback'] = "NA - Not enough satisfactory feedback"

        # Save to session state
        st.session_state["sat_clusters"] = sat_clusters
    else:   
        if "sat_elbow" not in st.session_state:
            st.session_state["sat_elbow"] = elbow_locator(st.session_state["sat_matrix"])

         # If elbow is not found set up default clusters
        if st.session_state["sat_elbow"] is None: 
            if len(st.session_state["df_satisfactory"]) < 20:
                st.session_state["sat_elbow"] = 2  
            else:
                st.session_state["sat_elbow"] = 3
            
        if "sat_clusters" not in st.session_state:
            st.markdown("### Clustering all Satisfacory Feedback")
            st.markdown("---")
            st.session_state["sat_clusters"] = get_cluster_labels(
                st.session_state["df_satisfactory"],
                st.session_state["df_original"],
                st.session_state["sat_matrix"],
                st.session_state["sat_elbow"],
                st.session_state["sat_vectorize"]
            )

    # Run clustering for dissatisfactory comments
    if "disat_elbow" not in st.session_state:
        st.session_state["disat_elbow"] = elbow_locator(st.session_state["disat_matrix"])

    if st.session_state["disat_elbow"] is None:  # If elbow is not found
        st.session_state["disat_elbow"] = 2  # Default to 2 clusters
        

    if "disat_clusters" not in st.session_state:
        st.markdown("### Clustering all Disatisfacory Feedback")
        st.markdown("---")
        st.session_state["disat_clusters"] = get_cluster_labels(
            st.session_state["df_disatisfactory"],
            st.session_state["df_original"],
            st.session_state["disat_matrix"],
            st.session_state["disat_elbow"],
            st.session_state["disat_vectorize"]
        )

else: 
   st.info("Once sentiment and feature exclusion have been applied, run clustering analysis.")


#________________________________________________________________________________________________
#                 5.  GIVE THE USER THE CHANCE TO DOWNLOAD THE NEW FILE AT THE END
#________________________________________________________________________________________________

# Ensure that the DataFrame is available at the end of the app for download
if "download" not in st.session_state:
    st.session_state["download"] = False

if st.button('Download Final Dataset'):
    st.session_state["download"] = True

if  st.session_state["download"]:

    #give the user the chance to save the new file
    file_name = st.text_input("Please type in the name you would like to save the file as: ", value="")

    # Add the 'cluster and feedback' for satisfactory reviews
    st.session_state['df_final'] = st.session_state['df_labeled'].copy()

    # Extract cluster columns and ensure they have the same index as the original DataFrame
    disat_cluster_map = st.session_state["disat_clusters"]['cluster and feedback']
    sat_cluster_map = st.session_state["sat_clusters"]["cluster and feedback"]
    
    # Combine the Series into one, maintaining their indices
    clusters_feedback = pd.concat([disat_cluster_map, sat_cluster_map])

    # Ensure the combined Series is correctly aligned with df_final's index

    st.session_state['df_final']['cluster and feedback'] = clusters_feedback.loc[st.session_state['df_final'].index]

    # Display the final DataFrame with the new column
    st.subheader("Here is a preview of the final dataset:")
    st.write(st.session_state['df_final'])
                 
    # Convert the DataFrame to CSV format for download
    csv = st.session_state["df_final"].to_csv(index=False)
    
    # Provide the download button
    if file_name:
        st.download_button(
            label="Download Updated DataFrame (with Sentiment & Cluster Labels)",
            data=csv,
            file_name= f"{file_name}.csv",
            mime="text/csv"
        )

else:
    st.info("Click 'Download' and Name Your File in Order to Save it.")