import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
df=pd.read_parquet('category_data.parquet')
print(df)

category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['category', 'total job count']
category_counts['total applications'] = df.groupby('category')['num_applications'].sum().values
category_counts['avg applications per job'] = category_counts['total applications'] / category_counts['total job count']



#dropping others category and F&B and wholesale trade  and personal care and general work and sales
category_counts = category_counts[category_counts['category'] != 'Others' ]
category_counts = category_counts[category_counts['category'] != 'F&B' ]
category_counts = category_counts[category_counts['category'] != 'Wholesale Trade' ]
category_counts = category_counts[category_counts['category'] != 'Personal Care / Beauty' ]
category_counts = category_counts[category_counts['category'] != 'General Work' ]
category_counts = category_counts[category_counts['category'] != 'Sales / Retail' ]


top_ten_total_jobs = category_counts.sort_values(by='total job count', ascending=False)
top_ten_total_jobs2 = top_ten_total_jobs[0:16]
top_ten_total_jobs3 = top_ten_total_jobs2[['category', 'total job count']]
print(top_ten_total_jobs3)
top_ten_total_applications = category_counts.sort_values(by='total applications', ascending=False)
top_ten_total_applications2 = top_ten_total_applications[0:16]
top_ten_total_applications3 = top_ten_total_applications2[['category', 'total applications']]
print(top_ten_total_applications3)

avg_applications=pd.merge(top_ten_total_jobs3,top_ten_total_applications3,on='category',how='inner')
avg_applications['avg applications per job']=avg_applications['total applications']/avg_applications['total job count']
avg_applications=avg_applications.sort_values(by='avg applications per job', ascending=False)
print(avg_applications)

st.title("Category Job Application Analysis")
st.write("Which category has the most competitive job market?")  

st.write("Which category has the most job postings?")

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax.bar(top_ten_total_jobs3['category'],top_ten_total_jobs2['total job count'], color='limegreen')
ax.set_xlabel('Category')
ax.set_ylabel('Total Job Count')
ax.set_title('Top 10 Job Categories by Total Job Count')
plt.xticks(rotation=45,ha='right')
st.pyplot(fig)
st.write("From the bar chart above,  it incidates that these job categories are the major hiring sectors in the market.")
st.write(top_ten_total_jobs3)
st.write("Which category receives the most applications?")

plt.figure(figsize=(10,10))
fig2, ax2 = plt.subplots()
ax2.bar(top_ten_total_applications3['category'],top_ten_total_applications2['total applications'], color='yellowgreen')
ax2.set_xlabel('Category')
ax2.set_ylabel('Total Applications')
ax2.set_title('Top 10 Job Categories by Total Applications')
plt.xticks(rotation=45,ha='right')
st.pyplot(fig2)
st.write(top_ten_total_applications3)
st.write("From this bar chart, it shows that these job categories are the most competitive sectors in the market, receiving the highest number of applications.")

st.write("With these data, we can summarise the data by calculating the average number of applications per job posting in each category to identify the most competitive job markets.")
plt.figure(figsize=(10,10))
fig3, ax3 = plt.subplots()
ax3.bar(avg_applications['category'],avg_applications['avg applications per job'], color='forestgreen')
ax3.set_xlabel('Category')
ax3.set_ylabel('Average Applications per Job')  
ax3.set_title('Average Applications per Job by Category')
plt.xticks(rotation=45,ha='right')
st.pyplot(fig3)

st.write(avg_applications)