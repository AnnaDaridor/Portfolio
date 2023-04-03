import pandas as pd
import numpy as np
import streamlit as st
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay, precision_recall_curve, average_precision_score, PrecisionRecallDisplay, precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Détection de fraude à la carte de crédit grâce au Machine Learning")

    # Fonction d'importation des données
    @st.cache(persist = True) # Pour ne pas recharger data à chaque fois
    def load_data():
        # Dataframes séparés pour pouvoir les importer sur Github
        df_1 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_1.csv")
        df_2 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_2.csv")
        df_3 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_3.csv")
        df_4 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_4.csv")
        df_5 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_5.csv")
        df_6 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_6.csv")
        df_7 = pd.read_csv("Streamlit/Détection_de_fraudes/creditcard_7.csv")
        data = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7]).drop('Unnamed: 0', axis = 1)
        return data

    # Affichage de la table de données
    df = load_data()
    df_sample = df.sample(100)
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeu de données : échantillon de 100 observations")
        st.write(df_sample)

    # Train/test split
    @st.cache(persist = True)
    def split(df):
        y = df['Class']
        X = df.drop('Class', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size = 0.2, 
            stratify = y, 
            random_state = 42
            )
        return X_train, X_test, y_train, y_test
        # Paramètre stratify = pour avoir la même distribution de 'class' que dans les données originelles

    X_train, X_test, y_train, y_test = split(df)    

    classifier = st.sidebar.selectbox(
        "Choisir un classificateur", 
        ("Random Forest", "SVM", "Logistic Regression")
        )

    # Analyse de la performance des modèles
    label_names = ['Transaction authentique', 'Transaction frauduleuse']
    def plot_perf(graphe):
        if "Matrice de confusion" in graphe:
            st.subheader("Matrice de confusion")
            matrice = confusion_matrix(y_test, y_pred)
            cmd = ConfusionMatrixDisplay(matrice, display_labels = label_names)
            cmd.plot()
            st.pyplot(bbox_inches='tight')
        if "Courbe ROC" in graphe:
            st.subheader("Courbe Roc")
            # Calculer la courbe ROC et l'AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            auc = roc_auc_score(y_test, y_scores)
            # Afficher la courbe ROC
            rcd = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=classifier)
            rcd.plot()
            st.pyplot(bbox_inches='tight')   
        if "Courbe precision-recall" in graphe:
            st.subheader("Courbe precision-recall")
            # Calculer la courbe PR et l'AP
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
            ap = average_precision_score(y_test, y_scores)
            # Afficher la courbe PR
            prd = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, estimator_name=classifier)
            prd.plot()
            st.pyplot(bbox_inches='tight')      
        

    # Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_estimators = st.sidebar.number_input(
            "Nombre d'arbres dans la forêt", 
            100, 1000, step = 10
            )
        max_depth = st.sidebar.number_input(
            "Profondeur maximale d'un arbre", 
            1, 20, step = 1
            )
        bootstrap = st.sidebar.radio(
            "Faire du bootstraping ?",
            (True, False)
            )   
        # Bootstrap = créer des échantillons d'entraînement aléatoires dans Random Forest, 
        # qui sont ensuite utilisés pour former des arbres de décision. 
        # Cela permet d'obtenir des prédictions plus robustes et moins sujettes au surapprentissage.

        graph_perf = st.sidebar.multiselect(
            "Graphiques de performance du modèle",
            ("Matrice de confusion", "Courbe ROC", "Courbe precision-recall")
        )

        if st.sidebar.button("Exécution", key = "classify"):
            st.subheader("Résultats du Random Forest")

            # Initialisation d'un Random Forest
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                bootstrap=bootstrap,
                random_state = 42
                )
            # Entraînement
            model.fit(X_train, y_train)

            # Prédictions
            y_pred = model.predict(X_test)
            
            # Calcul des probabilités de prédiction pour les données de test
            y_scores = model.predict_proba(X_test)[:, 1]

            # Métriques de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Affichage des métriques
            st.write("Accuracy :", accuracy.round(2))
            st.write("Précision :", precision.round(2))
            st.write("Recall :", recall.round(2))

            # Affichage des graphiques de performance
            plot_perf(graph_perf)


    # Support Vector Machine
    if classifier == "SVM":
        st.sidebar.subheader("Hyperparamètres du modèle")
        C = st.sidebar.number_input(
            "Paramètre de régularisation C", 
            0.01, 10.0
            )
        kernel = st.sidebar.radio(
            "Kernel", 
            ('rbf', 'linear')
            )
        gamma = st.sidebar.radio(
            "Valeur de gamma", 
            ('scale', 'auto')
            )
        graph_perf = st.sidebar.multiselect(
            "Graphiques de performance du modèle",
            ("Matrice de confusion", "Courbe ROC", "Courbe precision-recall")
        )

        if st.sidebar.button("Exécution", key = "classify"):
            st.subheader("Résultats du SVM")

            # Initialisation d'un SVM
            model = SVC(
                C = C, 
                kernel = kernel,
                gamma = gamma,
                random_state = 42
                )
            # Entraînement
            model.fit(X_train, y_train)

            # Prédictions
            y_pred = model.predict(X_test)

            # Métriques de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Affichage des métriques
            st.write("Accuracy :", accuracy.round(2))
            st.write("Précision :", precision.round(2))
            st.write("Recall :", recall.round(2))

            # Affichage des graphiques de performance
            plot_perf(graph_perf)


    # Régression Logistique
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparamètres du modèle")
        C = st.sidebar.number_input(
            "Paramètre de régularisation C", 
            0.01, 10.0
            )
        max_iter = st.sidebar.number_input(
            "Nombre maximal d'itérations", 
            100, 1000, step = 10
            )
        graph_perf = st.sidebar.multiselect(
            "Graphiques de performance du modèle",
            ("Matrice de confusion", "Courbe ROC", "Courbe precision-recall")
        )

        if st.sidebar.button("Exécution", key = "classify"):
            st.subheader("Résultats de la Régression Logistique")

            # Initialisation d'un Logistic Regression
            model = LogisticRegression(
                C = C, 
                max_iter = max_iter,
                random_state = 42
                )
            # Entraînement
            model.fit(X_train, y_train)

            # Prédictions
            y_pred = model.predict(X_test)

            # Métriques de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Affichage des métriques
            st.write("Accuracy :", accuracy.round(2))
            st.write("Précision :", precision.round(2))
            st.write("Recall :", recall.round(2))

            # Affichage des graphiques de performance
            plot_perf(graph_perf)        


if __name__ == '__main__':
    main()
