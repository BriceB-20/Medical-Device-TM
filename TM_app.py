import gzip
import streamlit as st
import pandas as pd
import datetime
import json
# import os
# from sympy import reduced
from top2vec import Top2Vec
from wordcloud import WordCloud
from matplotlib import pyplot as plt

# C:\Code\Medical-Device-Project\TM_app.py

# Load model & data
@st.experimental_singleton
def load_model():
    # path = 'C:/Code/Medical-Device-TM/'
    name = '2022-07-14_21_30_13_model[non-lemma,universal-sentence-encoder-large,deep_learn,min_count_10,ngram_True].json.gz'

    # try:
    with gzip.open(name, 'rb') as f:
        return Top2Vec.load(f)
    # except FileNotFoundError:
    #     with gzip.open(path + f'{name}', 'rb') as f:
    #         return Top2Vec.load(f)
    #     # return Top2Vec.load(path + f'{name}')

@st.cache
def load_data():
    def clean_data(data):
        json_file = json.load(data)
        date_cutoff = datetime.date(1989, 1, 1)
        df = pd.DataFrame.from_dict(json_file)
        df = df.drop_duplicates(subset='Abstract', keep=False) # Extract only items with abstracts (removes None duplicate)
        df = df.reset_index(drop=True)
        for i, item in enumerate(df["Date"]):  # Convert datetime strings back to objects
            df.at[i, 'Date'] = eval(item)
        df = df[df.loc[:, 'Date'] < date_cutoff]  # remove extraneous dates (from API issues)
        df = df.reset_index(drop=True)
        return df

    # path = 'C:/Code/Medical-Device-TM/'
    name = 'biomed_pubmed_data.json.gz'
    # try:
    with gzip.open(name, 'rb') as f:
        return clean_data(f)
    # except FileNotFoundError:
    #     with gzip.open(path + f'{name}', 'rb') as f:
    #         return clean_data(f)

def append_dict(data):
    # Create df with author info
    author_df = pd.DataFrame(columns=['Name', 'Affiliation'])
    for i, item in enumerate(data):
        author_df.at[i, 'Name'] = item['name']
        if item['affiliation'] is not None:
            author_df.at[i, 'Affiliation'] = item['affiliation']
        else:
            author_df.at[i, 'Affiliation'] = ''
    
    # Find longest affilitation string
    max_affiliation = 0
    for item in author_df['Affiliation']:
        if len(item) > max_affiliation:
            max_affiliation = len(item)
    
    # Add space to affiliation strings to match longest string
    for i, item in enumerate(author_df['Affiliation']):
        author_df.at[i, 'Affiliation'] = item + ' ' * (max_affiliation - len(item))

    return author_df

def create_frequency_dataframe(dataframe, default, specific, year_start=1976, year_stop=2000):
    # Create list of topic nums
    topic_sizes, topic_nums = model.get_topic_sizes(reduced=True)
    topic_columns = []
    if default == 'Top 5':
        topic_columns = topic_nums[:5]
    elif default == 'Top 10':
        topic_columns = topic_nums[:10]
    elif default == 'Bottom 5':
        topic_columns = topic_nums[-5:]
    elif default == 'Bottom 10':
        topic_columns = topic_nums[-10:]

    topic_columns= set(list(topic_columns) + [int(i) for i in specific.split() if len(i) > 0]) # Break input text into list of ints then merge lists
    
    # Format df for number of articles in each topic in a given year
    date_df = pd.DataFrame(index=range(year_start, year_stop + 1), columns=topic_columns)
    for col in date_df.columns:
        date_df[col].values[:] = 0  # Covert values from null to 0

    #Year/index num correspondence
    index_dict = {}
    for i, year in enumerate(date_df.index.values):
        index_dict[year] = i

    # Get doc_ids for each requested topic
    for topic in topic_columns:
        _, document_ids = model.search_documents_by_topic(topic_num=topic, reduced=True, num_docs=topic_sizes[topic]) # Works because topic number correspondes to its order in the topic size list
        for id in document_ids:
            if dataframe.at[id, 'Date'].year >= datetime.date(year_start,1,1).year and dataframe.at[id, 'Date'].year < datetime.date(year_stop + 1,1,1).year:
                date_df.at[dataframe.at[id, 'Date'].year, topic] += 1

    return date_df, topic_columns

def create_prevalence_dataframe(date_df):
    date_df = date_df.join(date_df.apply(sum, axis=1).reindex_like(date_df.index.to_series()).rename('Sums'))  # Sum the total number of articles in each year
    # Find the percent of total number of articles represented by each topic in each year
    for i, row in zip(date_df.index.values, date_df['Sums']):
        if row != 0:
            date_df.loc[i, :] = date_df.loc[i, :] / date_df.loc[i, 'Sums']
    date_df = date_df.drop(columns='Sums')
    return date_df

def create_plot(date_df, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for topic in date_df.columns:
        ax.plot(date_df.index.values, date_df.loc[:, topic].values, label=f'Topic {topic}')
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.grid(True)
    return fig

placeholder = st.empty()
placeholder.write("Loading. Please wait...")

model = load_model()
df = load_data()

placeholder.empty()


# Topic Reduction
# st.write('Performing topic reduction. This may take a few minutes')
# model.hierarchical_topic_reduction(50)

# App
mode = st.sidebar.selectbox('Select Mode',['Explore Topics', 'Explore Documents', 'Topic Prevalence'])

if mode == 'Explore Topics': # Variables associated with this section have the suffic 'et'
    with st.sidebar.form('Explore Topics', clear_on_submit=False):
        st.write('Input a search term(s) separated by a space and recieve the topics that are most semantically similar.')
        txt_et = st.text_area('Keyword(s)')
        num_et = st.number_input('Number of topics to return', 
                                min_value=1, 
                                max_value=model.get_num_topics(reduced=True), # Number of topics in the reduced model 
                                value=10)  # Default value is 10
        submitted_et = st.form_submit_button("Submit")
    if submitted_et:
        txt_et_list = txt_et.split()
        topic_words, word_scores, topic_scores, topic_nums = model.search_topics(txt_et_list, num_topics=num_et, reduced=True)
        for i, topic in enumerate(topic_nums):
            st.subheader('Topic Number: ' + str(topic))
            st.write('Search Cosine Similary: ' + str(topic_scores[i]))
            st.write('Topic Words:', ', '.join(topic_words[i][:10]))

            # Wordcloud
            new_dict = {}
            for ii, words in enumerate(topic_words[i]):
                new_dict[words] = word_scores[i][ii]
            st.image(WordCloud(width=850, height=250, 
                                background_color='white', 
                                prefer_horizontal=0.7).generate_from_frequencies(new_dict).to_image())

if mode == 'Explore Documents':
    with st.sidebar.form('Explore Documents', clear_on_submit=False):
        st.write('Input a search term(s) separated by a space and recieve the documents that are most semantical similar.')
        txt_ed = st.text_area('Keyword(s) or Phrase(s)')
        num_ed = st.number_input('Number of documents to return', 
                                min_value=1, 
                                max_value=50,
                                value=10)  # Default value is 10
        submitted_ed = st.form_submit_button("Submit")
    if submitted_ed:
        txt_ed_list = txt_ed.split()
        doc_scores, doc_ids = model.search_documents_by_keywords(txt_ed_list, num_docs=num_ed)
        st.container()
        for i, doc in enumerate(doc_ids):
            with st.container():
                st.subheader(df.iat[doc, 2])
                with st.expander('See more'):
                    st.write('Authors')
                    st.dataframe(append_dict(df.iat[doc, 5]))
                    st.write('Date:', f'{df.iat[doc, 6]}')
                    st.write('PMUID:', f'{df.iat[doc, 0]}')
                    st.write('Cosine Similarity:', str(doc_scores[i]))
                    st.write('Abstract:', f'{df.iat[doc, 3]}')

if mode == 'Topic Prevalence':
    with st.sidebar.form('Topic Prevalence', clear_on_submit=False):
        default = st.radio('Default Data', ['Top 5', 'Top 10', 'Bottom 5', 'Bottom 10', 'None'])
        specific = st.text_input('Topic Numbers', help='''Enter a series of topic numbers to graph. 
                                                        The numbers should be spearted by a space.
                                                        Topic numbers are assigned in order of decreasing prevalence.
                                                        In other words, Topic 0 is the most prevalent. 
                                                        Topic 1 is the second most prevalent, etc.''')
        year_start_tp, year_stop_tp = st.slider('Year Range', 
                                        min_value=df['Date'].min().year, 
                                        max_value=df['Date'].max().year, 
                                        value=[1976, 1988])
        submitted_ed = st.form_submit_button("Submit")
    if submitted_ed:
        frequency_df, topics = create_frequency_dataframe(dataframe = df, 
                                                        default = default, 
                                                        specific = specific, 
                                                        year_start = year_start_tp, 
                                                        year_stop = year_stop_tp)
        frequency_plot = create_plot(frequency_df, 'Number of Articles in Topic', 'Topic Frequency by Year')
        prevalence_df = create_prevalence_dataframe(frequency_df)
        prevalence_plot = create_plot(prevalence_df, ylabel='Proportion of Total Articles', title='Proportion of Articles in Topics by Year')
        
        st.pyplot(frequency_plot)
        st.dataframe(frequency_df.T)

        topic_words, _, topic_nums = model.get_topics(reduced=True)
        for topic in topics:
            st.write(f'Topic {topic}:', ', '.join(topic_words[topic][:10]))
        
        st.pyplot(prevalence_plot)
        st.dataframe(prevalence_df.T)