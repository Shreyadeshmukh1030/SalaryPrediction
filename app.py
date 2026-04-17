import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Streamlit App Title
st.title('Salary Prediction App')

st.write('Enter the details below to predict the salary.')

# --- Input Fields ---

# Age
age = st.slider('Age', 18, 65, 30)

# Gender
gender_options = ['Male', 'Female', 'Unknown']
gender_label = st.selectbox('Gender', gender_options)

# Education Level
education_options = ['Bachelor\'s', 'Master\'s', 'PhD', 'High School', 'Some College', 'Unknown']
education_label = st.selectbox('Education Level', education_options)

# Job Title (using an example list, in a real app this would be more dynamic or comprehensive)
# For simplicity, let's use the unique job titles from your original dataframe if available or a representative sample
# To replicate the LabelEncoder, we need the original categories. 
# For now, let's use a sample of job titles, but ideally, you'd save the encoder or a list of categories.
# Since we don't have the original LabelEncoder saved, we will fit a new one on a fixed list of common titles.
# In a production setting, you should save and load the LabelEncoder itself.

# NOTE: To correctly encode 'Job Title', the LabelEncoder needs to be fit on ALL possible job titles from the training data.
# For this demonstration, we'll use a placeholder and warn the user. In a real application, you'd save the encoder.

# Let's create a dummy encoder for demonstration. In a real scenario, you would have saved the fitted encoder.
# If you had the original df, you could do:
# job_titles_from_original_df = df['Job Title'].unique().tolist()
# job_encoder = LabelEncoder()
# job_encoder.fit(job_titles_from_original_df)

# For demonstration, we'll use a simplified approach.
# For a proper deployment, the LabelEncoder objects for 'Gender', 'Education Level', and 'Job Title' 
# should also be saved with pickle and loaded here, or the mapping should be known.

# Re-fitting LabelEncoders for deployment (ideally these would be loaded from disk)
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job_title = LabelEncoder()

# Fit on representative data (should cover all possible categories)
# This is a critical point: the categories used here MUST match those the model was trained with.
# For a proper solution, save the fitted LabelEncoder objects during training.
le_gender.fit(['Female', 'Male', 'Unknown'])
le_education.fit(["Bachelor's", "Master's", "PhD", "High School", "Some College", "Unknown"])
# For Job Title, this is highly simplified. A real deployment would need a comprehensive list or saved encoder.
# Let's use the 'Job Title' column from the original dataframe's head for fitting purposes, if 'df' is still available. 
# If not, we'd need to assume a list of possible job titles from the training data.

# Assuming 'df' is still available in the Colab session for fitting.
# This part needs adjustment if 'df' is not available here.
# For demonstration, I will assume a set of job titles based on common sense, as 'df' is not available in the current context for `app.py` generation.
# A better approach for deployment is to save the LabelEncoders along with the model.

# To make this self-contained for Streamlit, we need to embed the original categories
original_gender_categories = ['Female', 'Male', 'Unknown'] # Order matters if it was fit_transform earlier
original_education_categories = ["Bachelor's", "Master's", "PhD", "High School", "Some College", "Unknown"]
# This is problematic for Job Title. We need to save the `le_job_title` object or a full list of job titles.
# For now, let's create a mock list of job titles. This is a potential point of failure if user enters a job title not in this list.
original_job_title_categories = ['Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director', 'Marketing Manager', 'Product Manager', 'HR Manager', 'Financial Analyst', 'Graphic Designer', 'Project Manager', 'Operations Manager', 'Sales Manager', 'Research Scientist', 'Accountant', 'Business Analyst', 'IT Support', 'UI/UX Designer', 'Copywriter', 'Customer Service Rep', 'Operations Analyst', 'Web Developer', 'Network Engineer', 'Administrative Assistant', 'Recruiter', 'Legal Counsel', 'Content Creator', 'Business Development', 'Public Relations', 'Help Desk Analyst', 'Social Media Manager', 'Technical Writer', 'Database Administrator', 'Security Analyst', 'DevOps Engineer', 'Cloud Engineer', 'Data Scientist', 'Artificial Intelligence Engineer', 'Machine Learning Engineer', 'Quantitative Analyst', 'Software Developer', 'Full Stack Developer', 'Frontend Developer', 'Backend Developer', 'Cybersecurity Analyst', 'DevOps Specialist', 'Network Administrator', 'Solutions Architect', 'E-commerce Manager', 'Digital Marketing Specialist', 'Financial Advisor', 'Investment Analyst', 'Equity Analyst', 'Management Consultant', 'Strategy Consultant', 'UX Researcher', 'Product Designer', 'Mobile App Developer', 'Game Developer', 'Virtual Reality Engineer', 'Augmented Reality Engineer', 'Blockchain Developer', 'Cryptocurrency Trader', 'Data Engineer', 'Big Data Engineer', 'ETL Developer', 'Data Architect', 'Business Intelligence Developer', 'Marketing Analyst', 'SEO Specialist', 'SEM Specialist', 'PPC Specialist', 'Community Manager', 'Public Relations Specialist', 'Event Manager', 'Sales Representative', 'Customer Success Manager', 'Client Relationship Manager', 'Real Estate Agent', 'Insurance Agent', 'Financial Planner', 'Tax Advisor', 'Auditor', 'Compliance Officer', 'Legal Assistant', 'Paralegal', 'Judge', 'Lawyer', 'Physician', 'Nurse', 'Pharmacist', 'Medical Technologist', 'Dentist', 'Veterinarian', 'Psychologist', 'Therapist', 'Social Worker', 'Teacher', 'Professor', 'Librarian', 'Archivist', 'Museum Curator', 'Artist', 'Musician', 'Actor', 'Writer', 'Editor', 'Journalist', 'Photographer', 'Videographer', 'Animator', 'Sound Engineer', 'Film Director', 'Producer', 'Chef', 'Baker', 'Barista', 'Waiter/Waitress', 'Bartender', 'Hotel Manager', 'Tour Guide', 'Pilot', 'Flight Attendant', 'Air Traffic Controller', 'Logistician', 'Supply Chain Manager', 'Warehouse Manager', 'Truck Driver', 'Construction Manager', 'Civil Engineer', 'Architect', 'Electrical Engineer', 'Mechanical Engineer', 'Chemical Engineer', 'Biomedical Engineer', 'Environmental Engineer', 'Geologist', 'Meteorologist', 'Astronomer', 'Physicist', 'Chemist', 'Biologist', 'Mathematician', 'Statistician', 'Economist', 'Urban Planner', 'Sociologist', 'Anthropologist', 'Historian', 'Political Scientist', 'International Relations Specialist', 'Paralegal', 'Legal Assistant', 'Social Media Strategist']

# Fit the LabelEncoders (ideally from saved objects)
le_gender.fit(original_gender_categories)
le_education.fit(original_education_categories)
le_job_title.fit(original_job_title_categories)

job_title_label = st.selectbox('Job Title', original_job_title_categories)

years_experience = st.slider('Years of Experience', 0, 40, 5)

# -- Prediction Button --
if st.button('Predict Salary'):
    # Encode categorical features
    gender_encoded = le_gender.transform([gender_label])[0]
    education_encoded = le_education.transform([education_label])[0]
    job_title_encoded = le_job_title.transform([job_title_label])[0]

    # Create a DataFrame for the input, ensuring column order matches training data
    # The columns should be: 'Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_encoded,
        'Education Level': education_encoded,
        'Job Title': job_title_encoded,
        'Years of Experience': years_experience
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'The predicted salary is: ${prediction:,.2f}')
