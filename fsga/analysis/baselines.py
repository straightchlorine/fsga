"""Baseline feature selection methods for comparison.

Implements RFE, LASSO, Mutual Information, and Chi-squared feature selection
to compare against the GA approach.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE as SklearnRFE
from sklearn.feature_selection import (
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class BaselineSelector:
    """Base class for baseline feature selectors."""

    def __init__(self, random_state: int = 42):
        """Initialize baseline selector.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.selected_features_ = None
        self.scores_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineSelector":
        """Fit the selector on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self: Fitted selector
        """
        raise NotImplementedError

    def get_selected_features(self, k: int | None = None) -> np.ndarray:
        """Get indices of selected features.

        Args:
            k: Number of features to select (if None, use fitted selection)

        Returns:
            np.ndarray: Binary mask (1=selected) or indices
        """
        raise NotImplementedError

    def get_chromosome(self, n_features: int) -> np.ndarray:
        """Get binary chromosome representation of selected features.

        Args:
            n_features: Total number of features

        Returns:
            np.ndarray: Binary chromosome (1=feature selected)
        """
        chromosome = np.zeros(n_features, dtype=int)
        if self.selected_features_ is not None:
            chromosome[self.selected_features_] = 1
        return chromosome


class RFESelector(BaselineSelector):
    """Recursive Feature Elimination selector.

    Uses sklearn's RFE with a specified estimator to recursively remove
    features based on feature importance.
    """

    def __init__(
        self,
        estimator=None,
        n_features_to_select: int | None = None,
        step: int = 1,
        random_state: int = 42,
    ):
        """Initialize RFE selector.

        Args:
            estimator: Estimator with feature_importances_ or coef_ attribute
            n_features_to_select: Number of features to select (default: half)
            step: Number of features to remove at each iteration
            random_state: Random seed
        """
        super().__init__(random_state)
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.rfe_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFESelector":
        """Fit RFE on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self: Fitted selector
        """
        n_features_to_select = self.n_features_to_select or X.shape[1] // 2

        self.rfe_ = SklearnRFE(
            estimator=self.estimator,
            n_features_to_select=n_features_to_select,
            step=self.step,
        )
        self.rfe_.fit(X, y)

        self.selected_features_ = np.where(self.rfe_.support_)[0]
        self.scores_ = self.rfe_.ranking_  # Lower is better

        return self

    def get_selected_features(self, k: int | None = None) -> np.ndarray:
        """Get selected features.

        Args:
            k: Number of top features to select (if None, use fitted selection)

        Returns:
            np.ndarray: Indices of selected features
        """
        if k is None:
            return self.selected_features_
        else:
            # Select top k features by ranking
            return np.argsort(self.scores_)[:k]

    def __str__(self):
        return "RFE"


class LASSOSelector(BaselineSelector):
    """LASSO (L1 regularization) feature selector.

    Uses LassoCV to select features with non-zero coefficients.
    """

    def __init__(
        self, alphas: list[float] | None = None, cv: int = 5, random_state: int = 42
    ):
        """Initialize LASSO selector.

        Args:
            alphas: List of alpha values to try (if None, uses default range)
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        super().__init__(random_state)
        self.alphas = alphas
        self.cv = cv
        self.lasso_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LASSOSelector":
        """Fit LASSO on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self: Fitted selector
        """
        # Use LogisticRegression with L1 for classification
        if len(np.unique(y)) == 2:
            # Binary classification
            self.lasso_ = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0,
                random_state=self.random_state,
                max_iter=1000,
            )
        else:
            # Multiclass - use LassoCV for feature selection then LogisticRegression
            self.lasso_ = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0,
                random_state=self.random_state,
                max_iter=1000,
            )

        self.lasso_.fit(X, y)

        # Get non-zero coefficients
        if hasattr(self.lasso_, "coef_"):
            coef = self.lasso_.coef_
            if coef.ndim > 1:
                # Multiclass: use max across classes
                coef = np.abs(coef).max(axis=0)
            else:
                coef = np.abs(coef)

            self.scores_ = coef
            self.selected_features_ = np.where(coef > 0)[0]

        return self

    def get_selected_features(self, k: int | None = None) -> np.ndarray:
        """Get selected features.

        Args:
            k: Number of top features to select (if None, use non-zero coefs)

        Returns:
            np.ndarray: Indices of selected features
        """
        if k is None:
            return self.selected_features_
        else:
            # Select top k features by coefficient magnitude
            return np.argsort(-self.scores_)[:k]

    def __str__(self):
        return "LASSO"


class MutualInfoSelector(BaselineSelector):
    """Mutual Information feature selector.

    Uses sklearn's mutual_info_classif to rank features by MI with target.
    """

    def __init__(self, k: int = 10, random_state: int = 42):
        """Initialize Mutual Information selector.

        Args:
            k: Number of top features to select
            random_state: Random seed
        """
        super().__init__(random_state)
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInfoSelector":
        """Fit MI selector on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self: Fitted selector
        """
        self.scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        self.selected_features_ = np.argsort(-self.scores_)[: self.k]

        return self

    def get_selected_features(self, k: int | None = None) -> np.ndarray:
        """Get selected features.

        Args:
            k: Number of top features to select (if None, use fitted k)

        Returns:
            np.ndarray: Indices of selected features
        """
        k = k or self.k
        return np.argsort(-self.scores_)[:k]

    def __str__(self):
        return "MutualInfo"


class Chi2Selector(BaselineSelector):
    """Chi-squared feature selector.

    Uses sklearn's chi2 test to rank features. Requires non-negative features.
    """

    def __init__(self, k: int = 10):
        """Initialize Chi-squared selector.

        Args:
            k: Number of top features to select
        """
        super().__init__()
        self.k = k
        self.scaler = MinMaxScaler()  # Ensure non-negative features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Chi2Selector":
        """Fit Chi2 selector on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self: Fitted selector
        """
        # Scale to [0, 1] to ensure non-negativity
        X_scaled = self.scaler.fit_transform(X)

        self.scores_, _ = chi2(X_scaled, y)
        self.selected_features_ = np.argsort(-self.scores_)[: self.k]

        return self

    def get_selected_features(self, k: int | None = None) -> np.ndarray:
        """Get selected features.

        Args:
            k: Number of top features to select (if None, use fitted k)

        Returns:
            np.ndarray: Indices of selected features
        """
        k = k or self.k
        return np.argsort(-self.scores_)[:k]

    def __str__(self):
        return "Chi2"


class ANOVASelector(BaselineSelector):
    """ANOVA F-value feature selector.

    Uses sklearn's f_classif to rank features by ANOVA F-value.
    """

    def __init__(self, k: int = 10):
        """Initialize ANOVA selector.

        Args:
            k: Number of top features to select
        """
        super().__init__()
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANOVASelector":
        """Fit ANOVA selector on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self: Fitted selector
        """
        self.scores_, _ = f_classif(X, y)
        self.selected_features_ = np.argsort(-self.scores_)[: self.k]

        return self

    def get_selected_features(self, k: int | None = None) -> np.ndarray:
        """Get selected features.

        Args:
            k: Number of top features to select (if None, use fitted k)

        Returns:
            np.ndarray: Indices of selected features
        """
        k = k or self.k
        return np.argsort(-self.scores_)[:k]

    def __str__(self):
        return "ANOVA"


def get_baseline_selector(method: str, **kwargs) -> BaselineSelector:
    """Factory function to get baseline selector by name.

    Args:
        method: Selector name ('rfe', 'lasso', 'mi', 'chi2', 'anova')
        **kwargs: Additional arguments passed to selector

    Returns:
        BaselineSelector: Initialized selector

    Example:
        >>> selector = get_baseline_selector('rfe', n_features_to_select=10)
        >>> selector.fit(X_train, y_train)
        >>> selected = selector.get_selected_features()
    """
    selectors = {
        "rfe": RFESelector,
        "lasso": LASSOSelector,
        "mi": MutualInfoSelector,
        "mutual_info": MutualInfoSelector,
        "chi2": Chi2Selector,
        "anova": ANOVASelector,
    }

    method_lower = method.lower()
    if method_lower not in selectors:
        raise ValueError(
            f"Unknown baseline method: {method}. Available: {list(selectors.keys())}"
        )

    # Map 'k' parameter to the correct parameter name for each selector
    if "k" in kwargs:
        k_value = kwargs.pop("k")
        if method_lower == "rfe":
            kwargs["n_features_to_select"] = k_value
        elif method_lower in ["mi", "mutual_info", "chi2", "anova"]:
            kwargs["k"] = k_value
        # LASSO doesn't use k parameter, so we ignore it

    return selectors[method_lower](**kwargs)
