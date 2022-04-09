from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


def open_data(path):
    return pd.read_csv(path)


def main():
    cols_to_drop = ["emp_title", "annual_income_joint", "verification_income_joint", "debt_to_income_joint",
                    "application_type", "loan_amount", "term", "installment", "grade", "sub_grade", "issue_month",
                    "loan_status", "initial_listing_status", "disbursement_method", "balance", "paid_total", "paid_principal",
                    "paid_interest", "paid_late_fees"]
    df = open_data('loans_full_schema.csv')
    df = df[df["application_type"] == "individual"]  # filters out all joint applicants
    df = df.drop(columns=cols_to_drop)  # drop unused columns
    # factorize interest rates, data analysis showed rates weren't continuous
    prepped_labels = pd.factorize(df['interest_rate'].tolist())[0]

    # categorical features
    cat_attribs = ["state", "homeownership", "verified_income", "loan_purpose"]
    # all other features are numerical
    num_attribs = []
    for col in df.columns:
        if col not in cat_attribs:
            num_attribs.append(col)

    # replace null values with the median value
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    # Prep all variables in dataset
    prepped_df = full_pipeline.fit_transform(df)

    X_train, X_test, Y_train, Y_test = train_test_split(prepped_df, prepped_labels, test_size=0.2)

    per = SGDClassifier()

    per.fit(X_train, Y_train)

    X_train_prediction = per.predict(X_train)
    print("Stochastic Gradient Descent")
    print("Accuracy: {:.2%}".format(accuracy_score(Y_train, X_train_prediction)))

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, Y_train)
    print("Random Forest")
    print("Accuracy: {:.2%}".format(clf.score(X_train, Y_train)))
    print(clf.feature_importances_)


if __name__ == "__main__":
    main()
