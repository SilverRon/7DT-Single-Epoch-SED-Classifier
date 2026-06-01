import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

def sample_by_uid_group(df, n, uid_col='uid', random_state=42):
    """
    Sample data at the group (uid) level so that entire uid groups are selected until reaching n samples.
    """
    groups = df[uid_col].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(groups)
    selected_groups = []
    total_count = 0
    for g in groups:
        group_rows = df[df[uid_col] == g]
        selected_groups.append(g)
        total_count += len(group_rows)
        if total_count >= n:
            break
    sampled = df[df[uid_col].isin(selected_groups)]
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)
    return sampled

# Base class for all model experiments
class BaseExperiment:
    def __init__(self, X_train, X_test, y_train, y_test, label_encoder, params, eval_metrics_list, path_save, do_cv=False, groups=None):
        """
        Base class initializer.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.params = params
        self.eval_metrics_list = eval_metrics_list
        self.path_save = path_save
        self.do_cv = do_cv
        self.groups = groups
        self.model = None
        self.metrics_df = None
        self.creport = None
        self.y_pred = None

    def plot_confusion_matrix(self):
        """
        Plots and saves a confusion matrix with counts and row-normalized percentages.
        """
        import seaborn as sns
        cm = confusion_matrix(self.y_test, self.y_pred)
        labels = self.label_encoder.classes_
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)) * 100
        n_rows, n_cols = cm.shape
        combined_matrix = np.empty_like(cm, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                n = cm[i, j]
                p = cm_percent[i, j]
                combined_matrix[i, j] = f"{n:,}\n({p:.1f}%)"
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(cm_percent, annot=False, fmt="", cmap="Blues",
                         xticklabels=labels, yticklabels=labels,
                         cbar_kws={'label': '[%]'}, vmin=0, vmax=100)
        plt.xlabel("Predicted Label", fontsize=16)
        plt.ylabel("True Label", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(self.__class__.__name__.replace("Experiment", ""))
        plt.tight_layout()
        thresh = 50
        for i in range(n_rows):
            for j in range(n_cols):
                value = cm_percent[i, j]
                text_color = "white" if value > thresh else "black"
                ax.text(j+0.5, i+0.5, combined_matrix[i, j], ha='center', va='center', color=text_color, fontsize=14)
        plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_confusion_matrix.png"))

    def get_confusion_matrix(self):
        """
        Returns the confusion matrix and class labels.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        labels = self.label_encoder.classes_
        return cm, labels

    def _save_results(self, cv_scores=None):
        """
        Saves main metrics and classification report as CSV.
        """
        self.metrics_df.to_csv(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_metrics.csv"), index=False)
        pd.DataFrame.from_dict(self.creport).to_csv(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_classification_report.csv"))
        if cv_scores is not None:
            pd.DataFrame(cv_scores).to_csv(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_cv.csv"), index=False)

# --------------------------------------------------------------
class LightGBMExperiment(BaseExperiment):
    """
    LightGBM multi-class experiment.
    """
    def __init__(self, *args, **kwargs):
        import lightgbm as lgb
        self.lgb = lgb
        super().__init__(*args, **kwargs)

    def run(self):
        """
        Trains and evaluates a LightGBM model.
        """
        train_data = self.lgb.Dataset(self.X_train, label=self.y_train)
        test_data = self.lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)
        self.model = self.lgb.train(self.params, train_data, num_boost_round=1000,
                                    valid_sets=[train_data, test_data], valid_names=['train', 'test'])
        y_pred_prob = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        self.y_pred = np.argmax(y_pred_prob, axis=1)
        self._metrics_post()
        return self.model, self.metrics_df

    def _metrics_post(self):
        """
        Computes main classification metrics.
        """
        acc = accuracy_score(self.y_test, self.y_pred)
        self.creport = classification_report(self.y_test, self.y_pred, output_dict=True)
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        weighted_f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.metrics_df = pd.DataFrame({
            "model": [self.__class__.__name__.lower()],
            "accuracy": [acc], "precision_macro": [precision],
            "recall_macro": [recall], "f1_macro": [f1], "f1_weighted": [weighted_f1]
        })
        self._save_results(None)

    def plot_feature_importance(self, max_num=20):
        """
        Plots LightGBM feature importance (gain).
        """
        try:
            if hasattr(self.model, "feature_importance"):
                importances = self.model.feature_importance(importance_type="gain")
            else:
                raise AttributeError("feature_importance not found")
            feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(importances))]
            idx = np.argsort(importances)[::-1][:max_num]
            plt.figure(figsize=(8, 0.3*max_num+2))
            plt.barh(range(len(idx)), importances[idx][::-1])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1], fontsize=10)
            plt.xlabel("Importance", fontsize=14)
            plt.title(f"{self.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_feature_importance.png"))
        except Exception as e:
            print(f"Feature importance not supported for {self.__class__.__name__}: {e}")

# --------------------------------------------------------------
class CatBoostExperiment(BaseExperiment):
    """
    CatBoost multi-class experiment.
    """
    def __init__(self, *args, **kwargs):
        from catboost import CatBoostClassifier
        self.CatBoostClassifier = CatBoostClassifier
        super().__init__(*args, **kwargs)

    def run(self):
        """
        Trains and evaluates a CatBoost model.
        """
        self.model = self.CatBoostClassifier(**self.params, verbose=0, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self._metrics_post()
        return self.model, self.metrics_df

    def _metrics_post(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        self.creport = classification_report(self.y_test, self.y_pred, output_dict=True)
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        weighted_f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.metrics_df = pd.DataFrame({
            "model": [self.__class__.__name__.lower()],
            "accuracy": [acc], "precision_macro": [precision],
            "recall_macro": [recall], "f1_macro": [f1], "f1_weighted": [weighted_f1]
        })
        self._save_results(None)

    def plot_feature_importance(self, max_num=20):
        """
        Plots CatBoost feature importance.
        """
        try:
            importances = self.model.get_feature_importance()
            feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(importances))]
            idx = np.argsort(importances)[::-1][:max_num]
            plt.figure(figsize=(8, 0.3*max_num+2))
            plt.barh(range(len(idx)), importances[idx][::-1])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1], fontsize=10)
            plt.xlabel("Importance", fontsize=14)
            plt.title(f"{self.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_feature_importance.png"))
        except Exception as e:
            print(f"Feature importance not supported for {self.__class__.__name__}: {e}")

# --------------------------------------------------------------
class XGBExperiment(BaseExperiment):
    """
    XGBoost multi-class experiment.
    """
    def __init__(self, *args, **kwargs):
        import xgboost as xgb
        self.xgb = xgb
        super().__init__(*args, **kwargs)

    def run(self):
        """
        Trains and evaluates an XGBoost model.
        """
        self.model = self.xgb.XGBClassifier(**self.params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self._metrics_post()
        return self.model, self.metrics_df

    def _metrics_post(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        self.creport = classification_report(self.y_test, self.y_pred, output_dict=True)
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        weighted_f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.metrics_df = pd.DataFrame({
            "model": [self.__class__.__name__.lower()],
            "accuracy": [acc], "precision_macro": [precision],
            "recall_macro": [recall], "f1_macro": [f1], "f1_weighted": [weighted_f1]
        })
        self._save_results(None)

    def plot_feature_importance(self, max_num=20):
        """
        Plots XGBoost feature importance.
        """
        try:
            importances = self.model.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(importances))]
            idx = np.argsort(importances)[::-1][:max_num]
            plt.figure(figsize=(8, 0.3*max_num+2))
            plt.barh(range(len(idx)), importances[idx][::-1])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1], fontsize=10)
            plt.xlabel("Importance", fontsize=14)
            plt.title(f"{self.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_feature_importance.png"))
        except Exception as e:
            print(f"Feature importance not supported for {self.__class__.__name__}: {e}")

# --------------------------------------------------------------
class RFExperiment(BaseExperiment):
    """
    Random Forest multi-class experiment.
    """
    def __init__(self, *args, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self.RandomForestClassifier = RandomForestClassifier
        super().__init__(*args, **kwargs)

    def run(self):
        """
        Trains and evaluates a Random Forest model.
        """
        self.model = self.RandomForestClassifier(**self.params, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self._metrics_post()
        return self.model, self.metrics_df

    def _metrics_post(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        self.creport = classification_report(self.y_test, self.y_pred, output_dict=True)
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        weighted_f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.metrics_df = pd.DataFrame({
            "model": [self.__class__.__name__.lower()],
            "accuracy": [acc], "precision_macro": [precision],
            "recall_macro": [recall], "f1_macro": [f1], "f1_weighted": [weighted_f1]
        })
        self._save_results(None)

    def plot_feature_importance(self, max_num=20):
        """
        Plots Random Forest feature importance.
        """
        try:
            importances = self.model.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(importances))]
            idx = np.argsort(importances)[::-1][:max_num]
            plt.figure(figsize=(8, 0.3*max_num+2))
            plt.barh(range(len(idx)), importances[idx][::-1])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1], fontsize=10)
            plt.xlabel("Importance", fontsize=14)
            plt.title(f"{self.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_feature_importance.png"))
        except Exception as e:
            print(f"Feature importance not supported for {self.__class__.__name__}: {e}")

# --------------------------------------------------------------
class MLPExperiment(BaseExperiment):
    """
    MLP (Multi-layer Perceptron) multi-class experiment using scikit-learn.
    """
    def __init__(self, *args, **kwargs):
        from sklearn.neural_network import MLPClassifier
        self.MLPClassifier = MLPClassifier
        super().__init__(*args, **kwargs)

    def _metrics_post(self):
        """
        Calculates evaluation metrics and stores them in self.metrics_df.
        """
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score
        )

        y_true = self.y_test
        y_pred = self.y_pred

        # Calculate all required metrics
        metrics = {
            "f1_macro":     f1_score(y_true, y_pred, average="macro"),
            "f1_weighted":  f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy":     accuracy_score(y_true, y_pred)
        }
        import pandas as pd
        self.metrics_df = pd.DataFrame([metrics])

    def run(self):
        """
        Trains and evaluates an MLP model.
        """
        self.model = self.MLPClassifier(**self.params, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self._metrics_post()
        return self.model, self.metrics_df

    def plot_feature_importance(self, max_num=20):
        """
        Prints a warning (feature importance not supported for MLP).
        """
        print(f"Feature importance not supported for {self.__class__.__name__}")

# --------------------------------------------------------------
class AutoencoderExperiment(BaseExperiment):
    """
    Autoencoder-based multi-class classifier experiment using Keras.
    """
    def __init__(self, *args, **kwargs):
        from tensorflow import keras
        self.keras = keras
        super().__init__(*args, **kwargs)
        self.model = None

    def build_model(self, input_dim, num_classes, encoding_dim=32):
        """
        Builds and returns a simple autoencoder + classifier model.
        """
        Input = self.keras.layers.Input(shape=(input_dim,))
        # Encoder
        encoded = self.keras.layers.Dense(encoding_dim, activation='relu')(Input)
        # Decoder (for AE loss, not used for classification)
        decoded = self.keras.layers.Dense(input_dim, activation='linear')(encoded)
        # Classifier
        classifier_output = self.keras.layers.Dense(num_classes, activation='softmax')(encoded)

        model = self.keras.models.Model(inputs=Input, outputs=classifier_output)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def _metrics_post(self):
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score
        )
        y_true = self.y_test
        y_pred = self.y_pred

        metrics = {
            "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy":     accuracy_score(y_true, y_pred)
        }
        import pandas as pd
        self.metrics_df = pd.DataFrame([metrics])

    def run(self):
        """
        Trains autoencoder-based classifier.
        """
        # Define input and output
        X_train = self.X_train
        y_train = self.y_train
        X_test  = self.X_test
        y_test  = self.y_test

        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))

        # Build model
        self.model = self.build_model(input_dim, num_classes)
        
        # Train model
        self.model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        self.y_pred = self.model.predict(X_test).argmax(axis=1)
        self._metrics_post()
        return self.model, self.metrics_df

    def plot_feature_importance(self, max_num=20):
        print(f"Feature importance not supported for {self.__class__.__name__}")

# --------------------------------------------------------------
class TabNetExperiment(BaseExperiment):
    """
    TabNet classifier experiment.
    """
    def __init__(self, *args, **kwargs):
        from pytorch_tabnet.tab_model import TabNetClassifier
        self.TabNetClassifier = TabNetClassifier
        super().__init__(*args, **kwargs)

    def _metrics_post(self):
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score
        )
        y_true = self.y_test
        y_pred = self.y_pred

        metrics = {
            "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy":     accuracy_score(y_true, y_pred)
        }
        import pandas as pd
        self.metrics_df = pd.DataFrame([metrics])

    def run(self):
        """
        Trains and evaluates a TabNet model.
        """
        self.model = self.TabNetClassifier(**self.params, seed=42)
        self.model.fit(self.X_train.values, self.y_train,
                  eval_set=[(self.X_test.values, self.y_test)],
                  eval_metric=['accuracy'],
                  max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128)
        self.y_pred = self.model.predict(self.X_test.values)
        self._metrics_post()
        return self.model, self.metrics_df

    def plot_feature_importance(self, max_num=20):
        """
        Plots TabNet feature importance.
        """
        try:
            importances = self.model.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(importances))]
            idx = np.argsort(importances)[::-1][:max_num]
            plt.figure(figsize=(8, 0.3*max_num+2))
            plt.barh(range(len(idx)), importances[idx][::-1])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1], fontsize=10)
            plt.xlabel("Importance", fontsize=14)
            plt.title(f"{self.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_feature_importance.png"))
        except Exception as e:
            print(f"Feature importance not supported for {self.__class__.__name__}: {e}")

# --------------------------------------------------------------
class EBMExperiment(BaseExperiment):
    """
    Explainable Boosting Machine (EBM) multi-class experiment.
    """
    def __init__(self, *args, **kwargs):
        from interpret.glassbox import ExplainableBoostingClassifier
        self.EBMClassifier = ExplainableBoostingClassifier
        super().__init__(*args, **kwargs)

    def _metrics_post(self):
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score
        )
        y_true = self.y_test
        y_pred = self.y_pred

        metrics = {
            "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy":     accuracy_score(y_true, y_pred)
        }
        import pandas as pd
        self.metrics_df = pd.DataFrame([metrics])

    def run(self):
        """
        Trains and evaluates an EBM model.
        """
        self.model = self.EBMClassifier(**self.params)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self._metrics_post()
        return self.model, self.metrics_df

    def plot_feature_importance(self, max_num=20):
        """
        Plots EBM feature importance.
        """
        try:
            importances = self.model.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(importances))]
            idx = np.argsort(importances)[::-1][:max_num]
            plt.figure(figsize=(8, 0.3*max_num+2))
            plt.barh(range(len(idx)), importances[idx][::-1])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1], fontsize=10)
            plt.xlabel("Importance", fontsize=14)
            plt.title(f"{self.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_save, f"{self.__class__.__name__.lower()}_feature_importance.png"))
        except Exception as e:
            print(f"Feature importance not supported for {self.__class__.__name__}: {e}")


# =======================
# Example usage per model
# =======================
# Suppose you have:
# X_train, X_test, y_train, y_test, label_encoder, params, eval_metrics_list, path_save

# LightGBM
# lgbm_exp = LightGBMExperiment(X_train, X_test, y_train, y_test, label_encoder, params_lightgbm, eval_metrics_list, path_save)
# model, metrics = lgbm_exp.run()
# lgbm_exp.plot_confusion_matrix()
# lgbm_exp.plot_feature_importance()

# CatBoost
# cat_exp = CatBoostExperiment(X_train, X_test, y_train, y_test, label_encoder, params_catboost, eval_metrics_list, path_save)
# model, metrics = cat_exp.run()

# XGBoost
# xgb_exp = XGBExperiment(X_train, X_test, y_train, y_test, label_encoder, params_xgb, eval_metrics_list, path_save)
# model, metrics = xgb_exp.run()

# Random Forest
# rf_exp = RFExperiment(X_train, X_test, y_train, y_test, label_encoder, params_rf, eval_metrics_list, path_save)
# model, metrics = rf_exp.run()

# MLP
# mlp_exp = MLPExperiment(X_train, X_test, y_train, y_test, label_encoder, params_mlp, eval_metrics_list, path_save)
# model, metrics = mlp_exp.run()

# Autoencoder (custom code required)
# ae_exp = AutoencoderExperiment(X_train, X_test, y_train, y_test, label_encoder, params_autoencoder, eval_metrics_list, path_save)
# model, metrics = ae_exp.run()

# TabNet
# tabnet_exp = TabNetExperiment(X_train, X_test, y_train, y_test, label_encoder, params_tabnet, eval_metrics_list, path_save)
# model, metrics = tabnet_exp.run()

# EBM
# ebm_exp = EBMExperiment(X_train, X_test, y_train, y_test, label_encoder, params_ebm, eval_metrics_list, path_save)
# model, metrics = ebm_exp.run()

def plot_confusion_matrix(y_test, y_pred, labels):
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    # labels = label_encoder.classes_
    cm_percent = (cm / cm.sum(axis=1, keepdims=True)) * 100
    n_rows, n_cols = cm.shape
    combined_matrix = np.empty_like(cm, dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            n = cm[i, j]
            p = cm_percent[i, j]
            combined_matrix[i, j] = f"{n:,}\n({p:.1f}%)"
    fig = plt.figure(figsize=(12, 9))
    ax = sns.heatmap(cm_percent, annot=False, fmt="", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': '[%]'}, vmin=0, vmax=100)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(__class__.__name__.replace("Experiment", ""))
    plt.tight_layout()
    thresh = 50
    for i in range(n_rows):
        for j in range(n_cols):
            value = cm_percent[i, j]
            text_color = "white" if value > thresh else "black"
            ax.text(j+0.5, i+0.5, combined_matrix[i, j], ha='center', va='center', color=text_color, fontsize=14)
    return fig