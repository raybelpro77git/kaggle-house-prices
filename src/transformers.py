from sklearn.base import BaseEstimator, TransformerMixin


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
