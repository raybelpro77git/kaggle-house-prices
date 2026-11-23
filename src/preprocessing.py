import pandas as pd


# Identifiants Kaggle (colonne Id) des ventes anormales retirées du train :
# très grandes maisons (GrLivArea > 4000), qualité 10, quartier Edwards,
# vendues avant d'être terminées (Partial) au prix d'une maison moyenne.
OUTLIER_IDS = [524, 1299]


def remove_outliers(train):
    """Retire du train les ventes anormales (OUTLIER_IDS).

    À appeler sur le train brut (avec Id et SalePrice), avant d'extraire y.
    On s'appuie sur l'Id Kaggle et non sur la position de la ligne :
    le résultat ne dépend pas de l'ordre des lignes.
    """
    train = train.copy()
    train = train[~train["Id"].isin(OUTLIER_IDS)]
    return train.reset_index(drop=True)


def clean(all_X):
    """Nettoyage déterministe de train + test (aucune statistique apprise).

    Les imputations statistiques (médianes, modes) restent dans le Pipeline.
    """
    all_X = all_X.copy()

    # 1. Colonnes supprimées
    #    PoolQC : trop peu de piscines notées ; MiscFeature : MiscVal suffit ;
    #    GarageYrBlt : redondante avec YearBuilt ; Utilities : quasi constante
    all_X = all_X.drop(columns=["PoolQC", "MiscFeature", "GarageYrBlt", "Utilities"])

    # 2. NaN = absence de l'élément
    for col in ["Alley", "Fence", "FireplaceQu"]:
        all_X[col] = all_X[col].fillna("None")

    # 3. Placage : pas de type ET (surface à 0 ou vide) -> pas de placage
    #    (avant le remplissage de MasVnrArea, sinon les surfaces vides ne sont plus repérables)
    #    Les maisons avec une surface > 0 mais sans type restent NaN (ModeImputer)
    masque = (
        ((all_X["MasVnrArea"] == 0) | (all_X["MasVnrArea"].isna()))
        & (all_X["MasVnrType"].isna())
    )
    all_X.loc[masque, "MasVnrType"] = "None"
    all_X["MasVnrArea"] = all_X["MasVnrArea"].fillna(0)

    # 4. Garages : surface à 0 -> pas de garage
    #    (le masque n'inclut pas les surfaces vides : la maison concernée a un garage)
    cols_garage_cat = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    masque = (all_X["GarageArea"] == 0)
    all_X.loc[masque, cols_garage_cat] = "None"

    # 5. Sous-sols : surface à 0 -> pas de sous-sol
    cols_bsmt_cat = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    cols_bsmt_bath = ["BsmtFullBath", "BsmtHalfBath"]
    masque = (all_X["TotalBsmtSF"] == 0)
    all_X.loc[masque, cols_bsmt_cat] = "None"
    all_X.loc[masque, cols_bsmt_bath] = 0

    # 6. Sous-sol sans aucune information (surface totale vide) -> pas de sous-sol
    #    (masque calculé avant les affectations)
    cols_bsmt_num = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                     "BsmtFullBath", "BsmtHalfBath"]
    masque = (all_X["TotalBsmtSF"].isna())
    all_X.loc[masque, cols_bsmt_cat] = "None"
    all_X.loc[masque, cols_bsmt_num] = 0

    # 7. Functional : "typique" par défaut, selon la description
    all_X["Functional"] = all_X["Functional"].fillna("Typ")

    # 8. Codes catégoriels stockés en nombres -> texte (one-hot à l'encodage)
    for col in ["MSSubClass", "MoSold"]:
        all_X[col] = all_X[col].astype(str)

    return all_X


def add_features(all_X):
    """Ajoute les features retenues (voir le document de feature engineering).

    Calculs déterministes, appliqués après clean(). L'ordre compte :
    IsNew après le ramené à 0 des âges, IsRemodeled et HasPool avant
    la suppression des colonnes dont elles dépendent.
    """
    all_X = all_X.copy()

    # Surface totale : sous-sol + 1er étage + 2e étage
    all_X["TotalSF"] = all_X["TotalBsmtSF"] + all_X["1stFlrSF"] + all_X["2ndFlrSF"]

    # Âges au moment de la vente, ramenés à 0 pour les maisons vendues
    # avant d'être terminées (construction ou rénovation datée après la vente)
    all_X["HouseAge"] = (all_X["YrSold"] - all_X["YearBuilt"]).clip(lower=0)
    all_X["RemodAge"] = (all_X["YrSold"] - all_X["YearRemodAdd"]).clip(lower=0)

    # Rénovée : année de rénovation différente de l'année de construction
    # (avant la suppression des années)
    all_X["IsRemodeled"] = (all_X["YearBuilt"] != all_X["YearRemodAdd"]).astype(int)

    # Neuve à la vente (après le ramené à 0 des âges)
    all_X["IsNew"] = (all_X["HouseAge"] == 0).astype(int)

    # Nombre total de salles de bain (une demi-salle de bain compte pour 0,5)
    all_X["TotalBath"] = (
        all_X["FullBath"] + 0.5 * all_X["HalfBath"]
        + all_X["BsmtFullBath"] + 0.5 * all_X["BsmtHalfBath"]
    )

    # Présence d'un 2e étage et d'une piscine (avant la suppression de PoolArea)
    all_X["Has2ndFloor"] = (all_X["2ndFlrSF"] > 0).astype(int)
    all_X["HasPool"] = (all_X["PoolArea"] > 0).astype(int)

    # Colonnes remplacées : les années par les âges, PoolArea par HasPool
    all_X = all_X.drop(columns=["YearBuilt", "YearRemodAdd", "PoolArea"])

    return all_X
