import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def entrer_historique():
    history = deque(maxlen=14)
    print("Entrez les 14 derniers numéros gagnants :")
    for i in range(14):
        while True:
            try:
                numero = int(input(f"Entrez le numéro {i + 1} (0-36) : "))
                if 0 <= numero <= 36:
                    history.append(numero)
                    break
                else:
                    print("Veuillez entrer un numéro entre 0 et 36.")
            except ValueError:
                print("Veuillez entrer un nombre valide.")
    return history

def predire_numero(history, model):
    if len(history) < 14:
        return None
    X = np.array(history).reshape(1, -1)
    prediction = model.predict(X)[0]
    return prediction

def entrainer_modele(data):
    X = np.array([d[:-1] for d in data])
    y = np.array([d[-1] for d in data])

    # Create a Random Forest model
    rf = RandomForestClassifier()

    # Define a grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Print best hyperparameters
    print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")

    return best_model

def main():
    # Simulation de données historiques pour l'entraînement du modèle
    historical_data = []
    for _ in range(1000):
        historique = [np.random.randint(0, 37) for _ in range(14)]
        suivant = np.random.randint(0, 37)
        historique.append(suivant)
        historical_data.append(historique)

    model = entrainer_modele(historical_data)

    history = entrer_historique()
    prediction = predire_numero(history, model)
    if prediction is not None:
        print(f"Prédiction basée sur les 14 derniers numéros : {prediction}")

if __name__ == "__main__":
    main()
