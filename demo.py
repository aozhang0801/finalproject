import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix

@st.cache #use @st.cache decorator to cache the data, so it don't need to load data every panel update
def load_data(filename):  #function to load data
    df=pd.read_csv(filename)

    return df

def plot_confusion_matrix(cm_train,cm_test): #plot confusion matrix for train and test
    fig, ax = plt.subplots(1,2, figsize=(15, 10))
    ax[0].matshow(cm_train, cmap=plt.cm.GnBu)

    # 内部添加图例标签
    for x in range(len(cm_train)):
        for y in range(len(cm_train)):
            ax[0].annotate(cm_train[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    ax[0].set_ylabel('True Label')
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_title( "Confision Matrix for train data")

    ax[1].matshow(cm_test, cmap=plt.cm.Oranges)
    for x in range(len(cm_test)):
        for y in range(len(cm_test)):
            ax[1].annotate(cm_test[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    ax[1].set_ylabel('True Label')
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_title("Confision Matrix for test data")
    st.pyplot(fig)

def main_panel():  # main function, mainly used to layout widget and response for user input
    df = load_data("predictive_maintenance.csv") #df is a dataframe which contain all info in the excel

    st.title("Machine Predictive Maintenance Classification binary Vs multiclass :sunglasses:") # set title
    st.write("[Ao Zhang Github](https://github.com/aozhang0801/finalproject)")
    st.table(df.head(5))
    st.markdown("Show describe information:")
    st.table(df[["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]].describe())
    if st.sidebar.checkbox('Show EDA Process'):  # if the checkbox is selected
        st.markdown("")

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))  #target distribution
        targets = df["Target"].value_counts()
        ax0.bar(targets.index, height=targets.values, width=0.5, label="Target", tick_label=targets.index,
                facecolor='yellowgreen')
        for a, b in zip(targets.index, targets.values):
            ax0.text(a, b, '%.0f' % b, ha='center', va='bottom')
        ax0.set_title('target Distrbution')
        ax0.set_ylabel('count', fontdict={'weight': 'normal', 'size': 10})

        types = df[df["Failure Type"] != "No Failure"]["Failure Type"].value_counts()#Failure Type distribution
        ax1.set_title("Failure Type Distribution")
        ax1.bar(types.index, height=types.values, width=0.5, label="Failure Type", tick_label=types.index,
                facecolor='pink')
        for a, b in zip(types.index, types.values):
            ax1.text(a, b, '%.0f' % b, ha='center', va='bottom')
        fig.subplots_adjust(hspace=0.4)
        ax1.set_ylabel('count', fontdict={'weight': 'normal', 'size': 10})
        st.markdown('## Show target and Failure type distrbution:')
        st.pyplot(fig)

        fig, ax = plt.subplots(2, 2, figsize=(15, 10)) #Air temperature and process temperature distribution
        sns.distplot(df["Air temperature [K]"], label='Air temperature [K]', kde=True, ax=ax[0, 0])
        sns.distplot(df['Process temperature [K]'], label='Process temperature [K]', kde=True, ax=ax[0, 1])
        sns.boxplot(data=df, x='Air temperature [K]', ax=ax[1, 0])
        sns.boxplot(data=df, x='Process temperature [K]', ax=ax[1, 1])
        st.markdown('## Show temperature distrbution:')
        st.pyplot(fig)
        st.markdown('conclusion: There is no outliar in air temperature and process temperature')

        fig1, ax1 = plt.subplots(2, 3, figsize=(15, 10)) #Air temperature and Torque distribution
        sns.distplot(df["Rotational speed [rpm]"], label='Rotational speed [rpm]', kde=True, ax=ax1[0, 0])
        sns.distplot(df['Torque [Nm]'], label='Torque [Nm]', kde=True, ax=ax1[0, 1])
        sns.distplot(df['Tool wear [min]'], label='Tool wear [min]', kde=True, ax=ax1[0, 2])
        sns.boxplot(data=df, x='Rotational speed [rpm]', ax=ax1[1, 0])
        sns.boxplot(data=df, x='Torque [Nm]', ax=ax1[1, 1])
        sns.boxplot(data=df, x='Tool wear [min]', ax=ax1[1, 2])
        st.markdown('## Show other features distrbution:')
        st.pyplot(fig1)
        st.markdown('conclusion: rotational speed and torque have outliar.')

        fig2, ax2 = plt.subplots(1, 2, figsize=(15, 10)) #type distribution with Target =1 and target = 0
        ax2[0].set_title("Type distribution @ Target = 1")
        ax2[0].pie(df[df["Target"] == 1]["Type"].value_counts().values, explode=(0, 0, 0.1), labels=["L", "M", "H"],
                  autopct='%1.1f%%', startangle=150)
        ax2[1].set_title("Type distribution @ All")
        colors = ['r', 'y', 'b']
        ax2[1].pie(df["Type"].value_counts().values, explode=(0, 0, 0.1), labels=["L", "M", "H"], autopct='%1.1f%%',
                  startangle=150, colors=colors)
        st.markdown('## Show type distrbution:')
        st.pyplot(fig2)
        st.markdown('conclusion: the type has no influfernce with Failure.')

        st.markdown('## Show two features relationship:')
        fig3 = plt.figure(figsize=(15, 5))

        fig3 = plt.figure(figsize=(15, 5))
        plt.scatter(df[df["Target"] == 0]['Air temperature [K]'], df[df["Target"] == 0]['Process temperature [K]'],
                    alpha=0.5, marker="x", label="Target=0")
        plt.scatter(df[df["Target"] == 1]['Air temperature [K]'], df[df["Target"] == 1]['Process temperature [K]'],
                    alpha=0.75, marker="x", label="Target=1", c="r")
        plt.xlabel("Air temperature [K]")
        plt.ylabel("Process temperature [K]")
        plt.legend()
        st.pyplot(fig3)

        fig4 = plt.figure(figsize=(15, 5))
        sns.scatterplot(x='Rotational speed [rpm]', y='Torque [Nm]', hue='Target', alpha=0.85, data=df)
        st.pyplot(fig4)
        st.markdown('conclusion: lower rotational speed with larger Torque  and larger rotational speed with smaller Torque has more probability to failure.')

        fig5 = plt.figure(figsize=(15, 5))
        plt.scatter(df[df["Failure Type"] == "Power Failure"]['Air temperature [K]'],
                    df[df["Failure Type"] == "Power Failure"]['Process temperature [K]'],
                    alpha=0.5, marker="x", label="Failure Type=Power Failure")
        plt.scatter(df[df["Failure Type"] == "Tool Wear Failure"]['Air temperature [K]'],
                    df[df["Failure Type"] == "Tool Wear Failure"]['Process temperature [K]'],
                    alpha=0.75, marker="x", label="Failure Type=Tool Wear Failure", c="r")
        plt.scatter(df[df["Failure Type"] == "Overstrain Failure"]['Air temperature [K]'],
                    df[df["Failure Type"] == "Overstrain Failure"]['Process temperature [K]'],
                    alpha=0.75, marker="x", label="Failure Type=Overstrain Failure", c="g")
        plt.scatter(df[df["Failure Type"] == "Random Failures"]['Air temperature [K]'],
                    df[df["Failure Type"] == "Random Failures"]['Process temperature [K]'],
                    alpha=0.75, marker="x", label="Failure Type=Random Failures", c="Orange")
        plt.scatter(df[df["Failure Type"] == "Heat Dissipation Failure"]['Air temperature [K]'],
                    df[df["Failure Type"] == "Heat Dissipation Failure"]['Process temperature [K]'],
                    alpha=0.75, marker="x", label="Failure Type=Heat Dissipation Failure", c="Pink")

        plt.xlabel("Air temperature [K]")
        plt.ylabel("Process temperature [K]")
        plt.legend()
        st.pyplot(fig5)

        fig6 = plt.figure(figsize=(15, 5))
        sns.scatterplot(x='Rotational speed [rpm]', y='Torque [Nm]', hue='Failure Type', alpha=0.85, data=df)
        st.pyplot(fig6)

    df = df.drop("UDI", axis=1)
    df = df.drop("Product ID", axis=1)
    new_df = df.copy()
    le = LabelEncoder()
    new_df["Type"] = le.fit_transform(new_df["Type"])
    le1 = LabelEncoder()
    new_df["Failure Type"] = le1.fit_transform(new_df["Failure Type"])
    X = new_df[["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
                "Tool wear [min]"]]
    y = new_df["Target"]
    y_type = new_df["Failure Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X, y_type, random_state=0, test_size=0.25)

    if st.sidebar.checkbox('Feature Engineer'):
        st.markdown("## Data prepare: ")
        st.markdown("")
        st.markdown("1.  dropdown the columns of UDI and Product ID")
        st.markdown("2.  convert the category variables (Type,Failure Type) to number used sklearn LabelEncoder")
        st.markdown("3.  Split data to train and test with rate of 3:1")
        #st.markdown("4.  apply StandardScaler function to X data to scaler the predictors")
    if st.sidebar.checkbox('Modeling'):
        st.markdown("## Model for Target:")
        st.markdown("### LogisticRegression Model: ")
        lr = LogisticRegression().fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        fs_train = f1_score(y_train, y_train_pred)
        fs_test = f1_score(y_test, y_test_pred)
        c_train = confusion_matrix(y_train, y_train_pred)
        c_test = confusion_matrix(y_test, y_test_pred)
        st.markdown("f1-score for train data is %s, for test data is %s "%(fs_train,fs_test))
        plot_confusion_matrix(c_train,c_test)

        st.markdown("### DecisionTreeClassifier Model: ")
        dt = DecisionTreeClassifier().fit(X_train, y_train)
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)
        fs_train_dt = f1_score(y_train, y_train_pred)
        c_train = confusion_matrix(y_train, y_train_pred)
        fs_test_dt = f1_score(y_test, y_test_pred)
        c_test = confusion_matrix(y_test, y_test_pred)
        st.markdown("f1-score for train data is %s, for test data is %s " % (fs_train_dt, fs_test_dt))
        plot_confusion_matrix(c_train, c_test)

        st.markdown("## Model for Failure Type:")
        st.markdown("### LogisticRegression Model: ")
        lr = LogisticRegression().fit(X_train_type, y_train_type)
        y_train_pred_type = lr.predict(X_train_type)
        y_test_pred_type = lr.predict(X_test_type)
        fs_train_type = f1_score(y_train_type, y_train_pred_type,average="macro")
        fs_test_type = f1_score(y_test_type, y_test_pred_type,average="macro")
        c_train = confusion_matrix(y_train_type, y_train_pred_type)
        c_test = confusion_matrix(y_test_type, y_test_pred_type)
        st.markdown("f1-score for train data is %s, for test data is %s " % (fs_train_type, fs_test_type))
        plot_confusion_matrix(c_train, c_test)

        st.markdown("### DecisionTreeClassifier Model: ")
        dt = DecisionTreeClassifier().fit(X_train_type, y_train_type)
        y_train_pred_type = dt.predict(X_train_type)
        y_test_pred_type = dt.predict(X_test_type)
        fs_train_dt_type = f1_score(y_train_type, y_train_pred_type,average="macro")
        c_train = confusion_matrix(y_train_type, y_train_pred_type)
        fs_test_dt_type = f1_score(y_test_type, y_test_pred_type,average="macro")
        c_test = confusion_matrix(y_test_type, y_test_pred_type)
        st.markdown("f1-score for train data is %s, for test data is %s " % (fs_train_dt_type, fs_test_dt_type))
        plot_confusion_matrix(c_train, c_test)
    if st.sidebar.checkbox('Summary'):
        st.markdown("## Summary:")
        st.markdown("1. from the EDA process, Heat Dissipation Failure has relative of air temperature and proceess temperature,power failure has relative of rotatinal speed and torque")
        st.markdown("2. from model evalute result, decision tree is more effective compared with logistic regression")
        st.markdown("3. the target and faliure type is imbanlance, so for further discuss, we need some sample technology")


    st.write("### Reference:")
    st.write("data come from kaggle: https://www.kaggle.com/shivamb/machine-predictive-maintenance-classification")

main_panel()