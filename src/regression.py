#regression.py

#Import all needed ibraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():

    #Read the CSV file
    csvFile = "../data/results-survey779776.csv"

    df = pd.read_csv(csvFile)

    #Define features and goal columns
    feature_cols = ["G05Q18[1]", "G05Q18[2]", "G05Q18[3]", "G05Q18[4]"] #test data/column
    target_col = "G05Q18[5]"


    X = df[feature_cols]
    y = df[target_col]

    #Define and train model
    model = LinearRegression()

    model.fit(X, y)

    #Results 

    #Intercept
    print("Intercept:", model.intercept_)

    # Coeffiecent for every feature
    print("\Coefficient:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name}: {coef}")

    #Predict the model
    y_pred = model.predict(X)

    ergebnis_df = pd.DataFrame(
        {
            "y": y,
            "y_hat": y_pred,
        }
    )

    print("\Comparison: real vs. predicted Values:")
    print(ergebnis_df)

    plt.scatter(y, y_pred)
    plt.xlabel("Real Value (y)")
    plt.ylabel("Predicted Value")
    plt.title("Multiple Linear Regression real vs. predicted Values")

    min_wert = min(y.min(), y_pred.min())
    max_wert = max(y.max(), y_pred.max())
    plt.plot([min_wert, max_wert], [min_wert, max_wert])

    plt.show()


if __name__ == "__main__":
    main()

