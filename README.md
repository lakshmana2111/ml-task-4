# 🚗 EV Price Prediction Using Machine Learning

## 📌 Project Overview

This project uses **Machine Learning** and **Linear Regression** to analyze electric vehicle (EV) data and predict the price of an EV based on its driving range.

The project is implemented using **Python** in a **Jupyter Notebook / Google Colab** environment. The dataset contains information about different electric vehicles available in India, including their brand, model, price, range, power, and battery capacity.

---

## 🎯 Objectives

* Analyze an Electric Vehicle dataset.
* Explore EV specifications using Python.
* Handle missing values in the dataset.
* Use **EV Range** as the input feature.
* Predict **EV Price** using Linear Regression.
* Compare actual prices with predicted prices.
* Evaluate the Machine Learning model using standard regression metrics.
* Visualize the model's prediction performance.

---

## 📊 Dataset

The dataset contains **26 electric vehicle records** with **6 columns**.

| Column  | Description                 |
| ------- | --------------------------- |
| Brand   | Name of the EV manufacturer |
| Model   | EV model name               |
| Price   | Price of the vehicle        |
| Range   | Driving range of the EV     |
| Power   | Motor power                 |
| Battery | Battery capacity            |

### Sample Data

| Brand    | Model         | Price | Range | Power | Battery |
| -------- | ------------- | ----: | ----: | ----: | ------: |
| Maruti   | SuzukieVitara | 15.99 |   440 |   142 |      49 |
| Tata     | PunchEV       |  9.69 |   275 |    87 |      30 |
| Mahindra | XEV9e         | 21.90 |   542 |   228 |      59 |
| Mahindra | BE6           | 18.90 |   557 |   228 |      59 |
| MG       | WindsorEV     | 14.00 |   332 |   134 |      38 |

---

## 🛠️ Technologies Used

* **Python** – Programming language
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine Learning
* **Linear Regression** – Price prediction
* **Google Colab / Jupyter Notebook** – Development environment

---

## 🔄 Project Workflow

```text
EV Dataset
    ↓
Load Dataset
    ↓
Data Exploration
    ↓
Check Dataset Information
    ↓
Handle Missing Values
    ↓
Select Features
    ↓
Train-Test Split
    ↓
Linear Regression Model
    ↓
Make Predictions
    ↓
Compare Actual vs Predicted Prices
    ↓
Model Evaluation
    ↓
Visualization
```

---

## 🧹 Data Preprocessing

The project checks the structure and dimensions of the dataset before training the model.

Missing values in the important **Range** and **Price** columns are handled using `dropna()`.

```python
df = df.dropna(subset=["Range", "Price"])
```

After preprocessing, both columns contain no missing values.

---

## 🤖 Machine Learning Model

### Linear Regression

The project uses **Linear Regression** to predict EV prices based on driving range.

### Selected Features and Target

```python
X = df[["Range"]]
y = df["Price"]
```

The dataset is divided into training and testing sets using an **80:20 split** with `random_state=42`.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 📐 Regression Equation

The trained model produced approximately:

```text
Price = 0.1414 × Range − 34.3961
```

### Model Parameters

| Parameter |    Value |
| --------- | -------: |
| Slope     |   0.1414 |
| Intercept | -34.3961 |

The equation describes the fitted relationship between **driving range** and **EV price** in the dataset.

---

## 📈 Actual vs Predicted Values

The project compares the actual EV prices with the prices predicted by the Linear Regression model.

### Example Results

| Actual Price | Predicted Price |
| -----------: | --------------: |
|        21.49 |           41.68 |
|        17.29 |           31.78 |
|        15.99 |           27.82 |
|        72.50 |           33.90 |
|         3.25 |           -9.65 |
|         7.99 |            0.96 |

These results demonstrate the difference between actual and predicted EV prices.

> **Note:** The predicted values above are examples from the project. Negative predictions demonstrate a limitation of using a simple linear model for this dataset.

---

## 📊 Model Evaluation

The following regression metrics are used to evaluate the model:

| Metric                             | Description                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| **MAE** (Mean Absolute Error)      | Measures the average absolute difference between actual and predicted prices. |
| **MSE** (Mean Squared Error)       | Measures the average squared prediction error.                                |
| **RMSE** (Root Mean Squared Error) | Measures the square root of the mean squared error.                           |
| **R² Score**                       | Measures how much of the variation in price is explained by the model.        |

### Results

| Metric   |    Value |
| -------- | -------: |
| MAE      |  17.5081 |
| MSE      | 410.5502 |
| RMSE     |  20.2620 |
| R² Score |   0.2179 |

The model achieved an **R² score of approximately 0.218**, indicating that using only EV range as the predictor provides limited explanatory power for EV price in this dataset.

---

## 📉 Visualization

The project includes an **Actual vs Predicted EV Prices** scatter plot.

The visualization compares:

* **X-axis:** Actual Price
* **Y-axis:** Predicted Price
* **Reference line:** Helps compare predicted prices with actual prices.

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted EV Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()
```

---

## 📁 Project Structure

```text
EV-Price-Prediction/
│
├── Task_ML.ipynb
├── ev_car_India_dataset.csv
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Open the Project

Open `Task_ML.ipynb` using any of the following:

* Google Colab
* Jupyter Notebook
* JupyterLab
* VS Code

### 3. Add the Dataset

Make sure the following dataset is available in the project folder:

```text
ev_car_India_dataset.csv
```

### 4. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 5. Run the Notebook

Execute the notebook cells from top to bottom to reproduce the analysis and model results.

---

## 🚀 Future Improvements

The current model uses only **Range** to predict EV price. Prediction performance could potentially be improved by including additional features such as:

* Battery capacity
* Motor power
* Brand
* Vehicle model
* Multiple Linear Regression
* Random Forest Regression
* Decision Tree Regression
* Feature encoding and scaling
* Hyperparameter tuning

---

## 🎓 Learning Outcomes

Through this project, I learned how to:

* Load and inspect datasets using Pandas.
* Perform basic data preprocessing.
* Handle missing values.
* Select features and target variables.
* Split data into training and testing sets.
* Build a Linear Regression model.
* Generate predictions.
* Calculate regression evaluation metrics.
* Visualize actual vs predicted values.
* Interpret Machine Learning model performance.

---

## 📌 Conclusion

This project demonstrates how **Machine Learning** can be used to predict electric vehicle prices based on driving range.

Linear Regression provides a simple approach to understanding the relationship between EV range and price. However, the model's limited **R² score** shows that additional vehicle specifications and advanced Machine Learning algorithms may be required to achieve more accurate predictions.
