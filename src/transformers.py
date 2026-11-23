from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Remplit les NaN de target_col avec la médiane de son groupe (group_col),
    apprise uniquement sur les données passées à fit.
    Un groupe absent des données de fit reçoit la médiane globale."""

    def __init__(self, target_col="LotFrontage", group_col="Neighborhood"):
        # Paramètres uniquement (pas de calcul) : nécessaire pour clone() en CV
        self.target_col = target_col
        self.group_col = group_col

    def fit(self, X, y=None):
        # Apprend la médiane par groupe et la médiane globale (secours)
        self.medians_ = X.groupby(self.group_col)[self.target_col].median()
        self.global_median_ = X[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()  # ne modifie jamais les données d'origine

        # Médiane du groupe de chaque ligne (NaN si groupe inconnu)
        medianes_groupe = X[self.group_col].map(self.medians_)

        # Remplissage par la médiane du groupe, puis par la médiane globale
        X[self.target_col] = X[self.target_col].fillna(medianes_groupe)
        X[self.target_col] = X[self.target_col].fillna(self.global_median_)

        return X
class ModeImputer(BaseEstimator, TransformerMixin):
    """Remplit les NaN de target_col avec la valeur la plus fréquente,
    apprise uniquement sur les données passées à fit, en ignorant
    les valeurs de exclude_value.

    exclude_value : str ou liste de str
        Valeur(s) à ignorer pour le calcul du mode, ex. "None" (élément absent)
        ou ["None", "Unf"] (élément absent ou non aménagé).

    Hypothèse : seules les lignes à imputer sont encore NaN."""

    def __init__(self, target_col="MasVnrType", exclude_value="None"):
        # Paramètres uniquement (pas de calcul) : nécessaire pour clone() en CV
        self.target_col = target_col
        self.exclude_value = exclude_value

    def fit(self, X, y=None):
        # Un texte seul est mis dans une liste (list("None") découperait le texte
        # lettre par lettre) ; variable locale : on ne modifie pas le paramètre
        if isinstance(self.exclude_value, str):
            list_exclude_value = [self.exclude_value]
        else:
            list_exclude_value = self.exclude_value

        # Valeurs de target_col hors exclude_value (les NaN sont ignorés par mode())
        condition = (~X[self.target_col].isin(list_exclude_value))
        value = X.loc[condition, self.target_col]

        # Valeur la plus fréquente parmi ces valeurs
        self.mode_ = value.mode()[0]
        return self

    def transform(self, X):
        X = X.copy()  # ne modifie jamais les données d'origine

        # Remplissage des NaN restants par le mode appris
        X[self.target_col] = X[self.target_col].fillna(self.mode_)
        return X

class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Applique log1p (log(1 + x)) aux colonnes de cols.

    Sans état : fit n'apprend rien, il existe pour respecter
    le contrat du Pipeline (fit puis transform à chaque étape)."""
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy() 

        X[self.cols] = np.log1p(X[self.cols]) 

        return X
