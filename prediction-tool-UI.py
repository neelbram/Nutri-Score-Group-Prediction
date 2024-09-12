import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train the model (assuming the model is trained as per your code above)
@st.cache
def train_model():
    # Step 1: Load the training dataset
    train_data = pd.read_csv('train_data.csv')

    # Preprocess the dataset (train_data)
    train_data['description'] = train_data['description'].fillna('').str.lower()  # Fill NaN with empty string
    train_data['ingredients'] = train_data['ingredients'].fillna('').str.lower()  # Fill NaN with empty string

    # Step 2: Create features for all four decision tree models

    # Decision Tree 1: Based on common words in description
    keywords = ['organic', 'fresh', 'buttery', 'salty', 'rich', 'creamy']
    for word in keywords:
        train_data[word] = train_data['description'].apply(lambda x: 1 if word in x else 0)

    # Decision Tree 2: Based on presence of sugar, salt, and water in ingredients
    train_data['has_sugar'] = train_data['ingredients'].apply(lambda x: 1 if 'sugar' in x else 0)
    train_data['has_salt'] = train_data['ingredients'].apply(lambda x: 1 if 'salt' in x else 0)
    train_data['has_water'] = train_data['ingredients'].apply(lambda x: 1 if 'water' in x else 0)

    # Decision Tree 3: Based on 30 most common words in the ingredients
    common_words = ['organic', 'vitamin', 'garlic', 'whole', 'juice', 'rice', 'red', 'calcium', 'concentrate', 
                    'extract', 'sea', 'less', 'citric', 'color', 'cheese', 'potassium', 'yeast', 'cocoa', 
                    'yellow', 'syrup', 'artificial', 'palm', 'lecithin', 'chocolate', 'butter', 'salt', 
                    'water', 'wheat', 'flour', 'acid']
    for word in common_words:
        train_data[f'has_{word}'] = train_data['ingredients'].apply(lambda x: 1 if word in x else 0)

    # Decision Tree 4: Based on the number of ingredients
    train_data['num_ingredients'] = train_data['ingredients'].apply(lambda x: len(x.split(',')))

    # Step 3: Combine all features into one dataset
    all_features = keywords + ['has_sugar', 'has_salt', 'has_water'] + [f'has_{word}' for word in common_words] + ['num_ingredients']
    target = train_data['Group']  # Nutri-Score groups (A-E)

    # Step 4: Train the Random Forest classifier on train_data
    X_train = train_data[all_features]
    y_train = train_data['Group']
    random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    random_forest_model.fit(X_train, y_train)

    return random_forest_model, keywords, common_words

# Load the model
random_forest_model, keywords, common_words = train_model()

# Function to classify a new product
def classify_product(description, ingredients):
    description = description.lower()
    ingredients = ingredients.lower()

    # Extract features for the random forest model
    feature_1 = np.array([1 if word in description else 0 for word in keywords]).reshape(1, -1)
    has_sugar = 1 if 'sugar' in ingredients else 0
    has_salt = 1 if 'salt' in ingredients else 0
    has_water = 1 if 'water' in ingredients else 0
    feature_2 = np.array([has_sugar, has_salt, has_water]).reshape(1, -1)
    feature_3 = np.array([1 if word in ingredients else 0 for word in common_words]).reshape(1, -1)
    num_ingredients = len(ingredients.split(','))
    feature_4 = np.array([num_ingredients]).reshape(1, -1)

    # Combine all features
    combined_features = np.hstack([feature_1, feature_2, feature_3, feature_4])

    # Make prediction using the trained Random Forest model
    predicted_group = random_forest_model.predict(combined_features)[0]
    return predicted_group

# Streamlit user interface
st.title("Nutri-Score Group Prediction")

# Input fields for description and ingredients
description_input = st.text_input("Enter product description", "e.g., low fat organic milk")
ingredients_input = st.text_input("Enter product ingredients", "e.g., milk, vitamin D")

# Button to classify
if st.button("Predict Nutri-Score Group"):
    # Call the classification function and display the result
    predicted_group = classify_product(description_input, ingredients_input)
    st.success(f"The predicted Nutri-Score group is: {predicted_group}")
