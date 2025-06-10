import base64
import io
from django.http import JsonResponse
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import re

def slugify(text):
    return re.sub(r'[^\w\-]+', '', text.lower().replace(' ', '_'))

def generate_heatmap(request):
    dataset_encoded = request.session.get('dataset')
    if not dataset_encoded:
        return JsonResponse({'error': 'Dataset non disponible'}, status=400)

    decoded_csv = base64.b64decode(dataset_encoded).decode()
    data = pd.read_csv(io.StringIO(decoded_csv))

    heatmap_path = os.path.join('static', 'correlation_heatmap.png')
    if not os.path.exists('static'):
        os.makedirs('static')

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return JsonResponse({'heatmap_url': '/' + heatmap_path})

def model_train(request):
    # your code here
    return render(request, 'some_template.html')

def upload_file(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        csv_file = request.FILES['dataset']
        if not csv_file.name.endswith('.csv'):
            return render(request, 'upload.html', {'error': 'Le fichier doit être un CSV.'})

        try:
            content = csv_file.read().decode('utf-8')
        except UnicodeDecodeError:
            csv_file.seek(0)
            content = csv_file.read().decode('latin1')

        try:
            data = pd.read_csv(io.StringIO(content), sep=None, engine='python')
        except Exception as e:
            return render(request, 'upload.html', {'error': f"Erreur de lecture du CSV : {e}"})

        if data.empty:
            return render(request, 'upload.html', {'error': 'Le fichier CSV est vide.'})

        # Infos
        n_rows, n_cols = data.shape
        types = data.dtypes.to_dict()
        missing_values = data.isnull().sum().to_dict()
        description = data.describe(include='all').to_html(classes="table table-striped")
        data_preview = data.head(50).to_html(classes="table table-bordered")

        # Sauvegarder dans session
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_base64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        request.session['dataset'] = csv_base64

        context = {
            'data_preview': data_preview,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'types': types,
            'missing_values': missing_values,
            'description': description,
        }

        return render(request, 'upload.html', context)

    return render(request, 'upload.html')


def preprocess(request):
    dataset_encoded = request.session.get('dataset')
    if not dataset_encoded:
        return redirect('upload')

    decoded_csv = base64.b64decode(dataset_encoded).decode()
    data = pd.read_csv(io.StringIO(decoded_csv))

    missing_cols = [col for col in data.columns if data[col].isnull().any()]
    message = None

    if request.method == 'POST':
        for col in missing_cols:
            strategy = request.POST.get(f'strategy_{col}')
            if strategy == 'mean':
                data[col].fillna(data[col].mean(), inplace=True)
            elif strategy == 'most_frequent':
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                data[col].fillna(0, inplace=True)
            elif strategy == 'drop':
                data.drop(columns=[col], inplace=True)

        # Save updated dataset back to session
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_base64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        request.session['dataset'] = csv_base64

        missing_cols = [col for col in data.columns if data[col].isnull().any()]
        message = "Prétraitement appliqué avec succès."

    return render(request, 'preprocess.html', {
        'missing_cols': missing_cols,
        'message': message
    })


def feature_engineering(request):
    dataset_encoded = request.session.get('dataset')
    if not dataset_encoded:
        return redirect('upload')

    decoded_csv = base64.b64decode(dataset_encoded).decode()
    data = pd.read_csv(io.StringIO(decoded_csv))

    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    unique_values = {col: data[col].dropna().unique().tolist() for col in categorical_cols}

    if request.method == 'POST':
        norm_cols = request.POST.getlist('normalize_cols')
        norm_method = request.POST.get('normalization_method')
        dummy_cols = request.POST.getlist('dummy_cols')
        dummy_cols = [col for col in dummy_cols if col in data.columns]

        df_new = data.copy()

        if norm_method and norm_cols:
            if norm_method == 'minmax':
                scaler = MinMaxScaler()
            elif norm_method == 'standard':
                scaler = StandardScaler()
            else:
                scaler = None

            if scaler:
                df_new[norm_cols] = scaler.fit_transform(df_new[norm_cols])

        if dummy_cols:
            df_new = pd.get_dummies(df_new, columns=dummy_cols, drop_first=True)

        csv_buffer = io.StringIO()
        df_new.to_csv(csv_buffer, index=False)
        csv_base64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        request.session['dataset'] = csv_base64

        message = "Feature engineering appliqué avec succès."
        return render(request, 'feature_engineering.html', {
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'unique_values': unique_values,
            'message': message,
            'data_preview': df_new.head(20).to_html(classes="table table-bordered"),
        })

    return render(request, 'feature_engineering.html', {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'unique_values': unique_values,
        'data_preview': data.head(20).to_html(classes="table table-bordered"),
    })


# Fonctions pour régression linéaire
def F_regression(x, theta):
    return np.dot(x, theta)

def gradient_regression(X, y, theta):
    m = len(X)
    return (1/m) * X.T.dot(F_regression(X, theta) - y)

def gradient_descent(X, y, theta, alpha=0.01, iterations=1000):
    for _ in range(iterations):
        theta -= alpha * gradient_regression(X, y, theta)
    return theta

# Fonctions pour classification logistique
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def F_logistic(X, theta):
    return sigmoid(X.dot(theta))

def gradient_logistic(X, y, theta):
    m = len(X)
    h = F_logistic(X, theta)
    return (1/m) * X.T.dot(h - y)

def gradient_descent_logistic(X, y, theta, alpha=0.01, iterations=1000):
    for _ in range(iterations):
        theta -= alpha * gradient_logistic(X, y, theta)
    return theta


def standardize_features(X):
    """Retourne X normalisé, avec sa moyenne et son écart-type."""
    X = np.array(X, dtype=np.float64)   # <-- forcer float64
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # Gère le cas d'une seule colonne
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # éviter division par zéro
    return (X - mean) / std, mean, std


def standardize_input(input_array, mean, std):
    """Applique la même normalisation que celle des features d'entraînement."""
    input_array = np.array(input_array, dtype=np.float64)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    return (input_array - mean) / std

def prediction(request):
    dataset_encoded = request.session.get('dataset')
    if not dataset_encoded:
        return redirect('upload')

    decoded_csv = base64.b64decode(dataset_encoded).decode()
    data = pd.read_csv(io.StringIO(decoded_csv))
    columns = data.columns.tolist()

    context = {
        'columns': columns,
        'heatmap_url': None,
        'show_heatmap': False
    }

    if request.method == 'POST':
        action = request.POST.get('action')  # 'predict' ou 'show_heatmap'

        if action == 'show_heatmap':
            heatmap_path = os.path.join('static', 'correlation_heatmap.png')
            if not os.path.exists('static'):
                os.makedirs('static')

            plt.figure(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Matrice de corrélation')
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()

            context['heatmap_url'] = '/' + heatmap_path
            context['show_heatmap'] = True
            return render(request, 'prediction.html', context)

        elif action == 'predict':
            selected_features = request.POST.getlist('features')
            target = request.POST.get('target')
            model_type = request.POST.get('model_type')
            alpha = float(request.POST.get('alpha', 0.01))
            iterations = int(request.POST.get('iterations', 1000))
            test_size = float(request.POST.get('test_size', 0.2))
            visualisation = request.POST.get('show_visualisation') == 'on'

            input_values = []
            try:
                for feature in selected_features:
                    slug_name = slugify(feature)
                    val = request.POST.get(f'feature_value_{slug_name}')
                    if val is None or val.strip() == '':
                        raise ValueError(f"Valeur manquante pour {feature}")
                    input_values.append(float(val))
                input_array = np.array(input_values, dtype=np.float64).reshape(1, -1)
            except Exception as e:
                context.update({
                    'error': f"Erreur dans les valeurs d'entrée : {str(e)}",
                    'selected_features': selected_features,
                    'target': target,
                    'model_type': model_type,
                    'alpha': alpha,
                    'iterations': iterations
                })
                return render(request, 'prediction.html', context)

            X_raw = data[selected_features].values.astype(np.float64)
            if X_raw.ndim == 1:
                X_raw = X_raw.reshape(-1, 1)
            y = data[target].values

            # Standardisation
            X_std, mean, std = standardize_features(X_raw)
            X = np.hstack([np.ones((X_std.shape[0], 1)), X_std])
            input_std = standardize_input(input_array, mean, std)
            input_with_bias = np.insert(input_std, 0, 1, axis=1).flatten()

            # Split en train/test
            m = X.shape[0]
            split_index = int((1 - test_size) * m)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            theta = np.zeros(X.shape[1])

            if model_type == 'regression':
                theta = gradient_descent(X_train, y_train, theta, alpha, iterations)
                prediction_result = F_regression(input_with_bias, theta)
                y_pred = F_regression(X_test, theta)
                r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
                # 🔽 Visualisation Prédictions vs Valeurs Réelles
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7, label='Prédictions')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Idéal')
                plt.xlabel("Valeurs réelles")
                plt.ylabel("Prédictions")
                plt.title("Prédictions vs Réalité (Set de Test)")
                plt.legend()
                plt.grid(True)
                vis_r2_path = os.path.join('static', 'r2_scatter.png')
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig(vis_r2_path)
                plt.close()
                context['r2_plot_url'] = '/' + vis_r2_path

                context.update({
                    'prediction': prediction_result,
                    'r2_score': r2,
                    'model_type': 'Régression Linéaire'
                })

            elif model_type == 'classification':
                y_bin = np.array(y)
                if not set(np.unique(y_bin)).issubset({0, 1}):
                    context.update({
                        'error': "Pour la classification, la cible doit être binaire (0/1).",
                        'selected_features': selected_features,
                        'target': target,
                        'model_type': model_type,
                        'alpha': alpha,
                        'iterations': iterations,
                        'input_values': input_values
                    })
                    return render(request, 'prediction.html', context)

                theta = gradient_descent_logistic(X_train, y_train, theta, alpha, iterations)
                prob = F_logistic(input_with_bias, theta)
                prediction_result = int(prob >= 0.5)
                accuracy = np.mean((F_logistic(X_test, theta) >= 0.5) == y_test)
                prediction_result = "True: " + target if prediction_result == 1 else "False: Not " + target

                context.update({
                    'prediction': prediction_result,
                    'probability': prob,
                    'accuracy': accuracy,
                    'model_type': 'Classification Logistique'
                })

            # Visualisation avec toutes les données du CSV
            if visualisation and len(selected_features) in [1, 2]:
                fig = plt.figure()
                X_vis = X[:, 1:]  # X complet (avec biais retiré), pas X_test

                if model_type == 'regression':
                    y_pred_all = F_regression(X, theta)
                    plt.scatter(X_vis[:, 0], y, color='blue', label='Valeurs réelles')
                    plt.scatter(X_vis[:, 0], y_pred_all, color='red', label='Valeurs prédites')
                    plt.title("Régression Linéaire (ensemble complet)")
                    plt.xlabel(selected_features[0])
                    plt.ylabel("Valeur cible")
                    plt.legend()

                elif model_type == 'classification' and len(selected_features) == 2:
                    plt.xlabel(selected_features[0])
                    plt.ylabel(selected_features[1])
                    plt.plot(X_vis[:, 0], X_vis[:, 1], 'x', label='Données CSV')

                    theta0, theta1, theta2 = theta[0], theta[1], theta[2]
                    x_vals = np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 100)
                    y_vals = -(theta1 * x_vals + theta0) / theta2

                    plt.plot(x_vals, y_vals, color='magenta',
                            label=r"Frontière de décision : $G(X) = a_1 x_1 + a_2 x_2 + b = 0$")
                    plt.title("Classification Logistique (ensemble complet)")
                    plt.legend()

                plt.tight_layout()
                vis_path = os.path.join('static', 'prediction_visualisation.png')
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig(vis_path)
                plt.close()
                context['plot_url'] = '/' + vis_path
                context['show_heatmap'] = False


            context.update({
                'columns': columns,
                'selected_features': selected_features,
                'target': target,
                'alpha': alpha,
                'iterations': iterations,
                'test_size': test_size,
                'input_values': input_values
            })

            return render(request, 'prediction.html', context)

    return render(request, 'prediction.html', context)
