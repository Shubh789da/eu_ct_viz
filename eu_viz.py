#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import plost
from time import sleep
from stqdm import stqdm
from bs4 import BeautifulSoup
from urllib.parse import quote

last_page = None
total_ct = None
i = None

@st.cache_data
def create_dataframe(query):
    
    global last_page, total_ct, i
    
    def convert_with_comma_to_int(number_str):
        if ',' in number_str:
            number_str = number_str.replace(",", "")
        return int(number_str)

    full_query = quote(query)
    base_url_web = 'https://www.clinicaltrialsregister.eu/ctr-search/search?query={}'
    full_url_web = base_url_web.format(full_query,1)


    response = requests.get(full_url_web)
    soup = BeautifulSoup(response.content, 'html.parser')


    outcome_div = soup.find('div', class_='outcome')
    total_results_text = outcome_div.text.strip()
    total_ct_str = total_results_text.split('result(s) found for: ')[0]
    total_ct= convert_with_comma_to_int(total_ct_str)

    last_page_str = total_results_text.split()[-1].split('.')[-2]
    last_page= convert_with_comma_to_int(last_page_str)
    df_final = pd.DataFrame()  # Initialize an empty DataFrame to store the final data

    # Iterate through each line in the text data
    for i in stqdm(range(1, last_page + 1),desc=f"We are loading your total {last_page} search pages. It may take time currently we are loading : ", mininterval=1):
        sleep(0.5)
        base_url = "https://www.clinicaltrialsregister.eu/ctr-search/rest/download/summary?query={}&page={}&mode=current_page"
        full_url = base_url.format(full_query, i)
        response = requests.get(full_url)
        text_data = response.text

        # Split the text data by lines
        lines = text_data.strip().split('\r\n')

        # Initialize variables to store record information
        records = []
        record = {}
        current_key = None
        last_key = None
        keys=['EudraCT Number', 'Sponsor Protocol Number', 'Sponsor Name',
       'Full Title', 'Start Date', 'Medical condition', 'Disease',
       'Population Age', 'Gender', 'Trial protocol','Link']

        # Iterate through each line in the text data
        for line in lines:
            # Check if the line contains a key-value pair
            if ':' in line:
                last_key = current_key if current_key in keys else last_key  # Update the last_key variable
                key, value = line.split(':', 1)
                current_key = key.strip()

                if current_key == "Disease":
                    # Process the "Disease" value to extract the required information
                    value = value.split(", Term: ")[-1].split(", Level: ")[0]
                    if current_key in record:
                        record[current_key].append(value)
                    else:
                        record[current_key] = [value]
                        
                elif current_key not in keys:
                    record[last_key] += " " + line.strip()
                    
                else:
                    record[current_key] = value.strip()
            else:

                if current_key in keys:
                    record[current_key] += " " + line.strip()
                    
                elif (line != (None or "") and last_key in keys) :
                    record[last_key] += " " + line.strip()
            # Check if the line contains "Link", which indicates the end of a record
            if current_key == "Link":
                # Append the current record to the records list
                records.append(record)
                # Reset the record and current_key variables for the next record
                record = {}
                current_key = None


        df = pd.DataFrame.from_records(records)
        df_final = pd.concat([df_final, df], ignore_index=True)
        df_final['Start Date']=pd.to_datetime(df_final['Start Date'], errors = 'coerce')
        df_final['Phase'] = df_final['Full Title'].str.extract(r'(Phase\s[\w\s*?]+?\b)', flags=re.IGNORECASE)

        Ph_1 = ["phase 1",'Phase 1','PHASE 1','PHASE 1A','Phase 1a,PHASE 1B','phase 1b','Phase 1b','phase I','Phase I','PHASE I',
                'phase Ib','Phase Ib','Phase IB','PHASE IB','PHASE Ib']
        Ph_2 = ['PHASE 2','phase 2','Phase 2','phase 2a','Phase 2a','Phase 2b','PHASE II','phase II','Phase II','phase ii',
                'Phase IIa','phase IIa','Phase IIA','Phase IIb','PHASE IIb','phase IIb','phase ll']
        Ph_3 = ['Phase 3','PHASE 3','phase 3','Phase 3b','Phase III','phase III','PHASE III','phase IIIb',
                'PHASE IIIB','Phase IIIb','Phase IIIB']
        Ph_4 = ['Phase 4','phase 4','PHASE 4','Phase IV','PHASE IV','phase IV']


        df_final['CT_Phase'] = df_final['Phase'].apply(lambda x: 
            'Phase 1' if x in Ph_1 else (
                'Phase 2' if x in Ph_2 else (
                    'Phase 3' if x in Ph_3 else (
                        'Phase 4' if x in Ph_4 else None
                    )
                )
            )
        )


        
    return df_final

if 'query' not in st.session_state:
    st.session_state.CONNECTED = False
    st.session_state.query = ''

def _connect_form_cb(connect_status):
    st.session_state.CONNECTED = connect_status

def display_db_connection_menu():
    with st.form(key="connect_form"):
        st.text_input('Enter the condition', help='Click on search, pressing enter will not work', key='query')
        submit_button = st.form_submit_button(label='Search', on_click=_connect_form_cb, args=(True,))
        if submit_button:
            if not st.session_state.query:
                st.write("Please enter a condition")
                st.stop()
                
display_db_connection_menu()

if st.session_state.query:

    df = create_dataframe(st.session_state.query)
    
    #Heading for sidebar
    st.sidebar.header('Europe CT Dashboard `version 0.1`')
    
    #selecting the study by gender in ct
#     df['Gender']=df.loc[:,'Gender'].apply( lambda x: 'N/A' if len(x)==0 else ' '.join(map(str,x))) 
    options_st = df['Gender'].unique().tolist()
    options_st.insert(0, "All")
    selected_options_str = st.sidebar.selectbox('Select gender in CT?',options= options_st)
    
    if selected_options_str == 'All':
        filtered_df = df
    else:      
         #Convert selected_options_str back to a list
#         selected_options = selected_options_str.split()
        filtered_df= df[df.Gender.isin([selected_options_str])]
        pass
        if filtered_df.empty:
            st.write("No studies found for the selected options.")
            st.stop()

    # Slider for selecting year and month
  
    st.sidebar.subheader('Start Year of CT')
    years = filtered_df['Start Date'].dt.year.unique()
    if int(min(years))< int(max(years)):
        selected_year_range = st.sidebar.slider('Select Year Range', min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))), key='slider_year')
    else:
        st.sidebar.write(f"The only year available is: {int(max(years))}")    

    if int(min(years))< int(max(years)):
        # Filter the DataFrame based on the selected dates
        filtered_df = filtered_df[(filtered_df['Start Date'].dt.year >= selected_year_range[0]) & (filtered_df['Start Date'].dt.year <= selected_year_range[1])]
        # filtered_df['StartDate'] = filtered_df['StartDate'].dt.strftime('%Y-%m')
        filtered_df['CT_Phase']=filtered_df['CT_Phase'].fillna('N/A')
    else:
         filtered_df = filtered_df[(filtered_df['Start Date'].dt.year == (max(years)))]



    #Data for pie chart
#     filtered_df.loc[:,'Phase_str'] = filtered_df.loc[:,'Phase'].apply(lambda x: 'N/A' if len(x) == 0 else ' '.join(map(str, x)))
    filtered_df_pie=filtered_df.groupby("CT_Phase")['EudraCT Number'].count().rename('count_phase').reset_index()


    #Select the Phase for pie chart
    options = filtered_df_pie['CT_Phase'].unique().tolist()
    selected_options = st.sidebar.multiselect('Which phase do you want to analyse?',options)



    st.sidebar.markdown('''
    ---
    Created with ❤️ by [Shubhanshu](https://www.linkedin.com/in/shubh789/).
    ''')
    
    #data for side bars
    filtered_df["StartYear"]=filtered_df['Start Date'].dt.year
    if len(selected_options) == 0:
        filtered_df_lc=filtered_df.groupby('StartYear')['EudraCT Number'].count().rename('Nos_CT').reset_index()
        
    else:
        filtered_df_lc_pie = filtered_df[filtered_df.CT_Phase.isin(selected_options)]
        filtered_df_lc = filtered_df_lc_pie.groupby('StartYear')['EudraCT Number'].count().rename('Nos_CT').reset_index()


    # Row A
    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric("Nos. of studies", filtered_df_lc.Nos_CT.sum())

    #Nos. of ongoing studies
    ongoing_count = (
        filtered_df['Trial protocol'][filtered_df['Trial protocol'].str.contains('Ongoing')].count()
        if len(selected_options) == 0
        else filtered_df_lc_pie['Trial protocol'][filtered_df_lc_pie['Trial protocol'].str.contains('Ongoing')].count()
    )

    col2.metric("Nos. Ongoing CT", ongoing_count)

    #Nos. of completed studies
    completion_count = (
      filtered_df['Trial protocol'][filtered_df['Trial protocol'].str.contains('Completed')].count()
        if len(selected_options) == 0
        else filtered_df_lc_pie['Trial protocol'][filtered_df_lc_pie['Trial protocol'].str.contains('Completed')].count()
    )
    col3.metric("Nos. of Trials completed", completion_count)

    #row B
    c1,c2=st.columns((7,3))

    with c1:
        st.markdown('### Clinical trials per year')
        st.line_chart(filtered_df_lc, x = 'StartYear', y = 'Nos_CT',)

    if len(selected_options) == 0:
        filtered_pie = filtered_df_pie  # No filtering required, keep all data
    else:
        filtered_pie = filtered_df_pie[filtered_df_pie['CT_Phase'].isin(selected_options)]


    with c2:
        st.markdown('### Phase distribution')
        plost.donut_chart(
            data=filtered_pie,
            theta='count_phase',
            color='CT_Phase',
            legend='CT_Phase',
            use_container_width=True)

     

    dataExploration = st.container()

    with dataExploration:
#       st.title('Clinical trials data')
      st.subheader('Sample data')
#       st.header('Dataset: Clinical trials of', st.session_state.text)
      st.markdown('I found this dataset at... https://www.clinicaltrialsregister.eu/')
      st.markdown('**It is a sample of 100 rows from the dataset**')
#       st.text('Below is the sample DataFrame')
      st.dataframe(filtered_df.head(100))
        


