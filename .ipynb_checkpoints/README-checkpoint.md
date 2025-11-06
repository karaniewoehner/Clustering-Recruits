# Recruit to Current Player Finder

A Streamlit web app that compares **current Davidson College football players** to **recruits** based on athletic testing data.  
The model uses **KNN imputation**, **scaling**, **PCA**, and **K-Means clustering** to group athletes and find the most similar recruits to each player. This was created as a coding project for Cat Stats commissioned by the Davidson College Football Team.

---

## Features
- Able to upload your own **Test Numbers** sheet (.csv or .xlsx).
- Select a current player and view the **top similar recruits**.
- Filter results by position.
- View predicted **cluster assignment** and distance scores.
- Optionally **download** the matching results as a CSV file.

---

## Methodology
1. The underlying model (`recruit_model.joblib`) was trained on historical recruit combine data.  
2. Each player’s metrics are standardized and projected into PCA space.  
3. The app uses **Nearest Neighbors** to find recruits closest to the selected current player.  
4. All processing steps (imputation → scaling → PCA → clustering) are handled through a saved **`scikit-learn` Pipeline**.

---

## Project Structure
