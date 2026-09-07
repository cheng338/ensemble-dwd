"""Fit the binary ensemble on the bundled digits dataset; no download needed."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_dwd import EnsembleKernelDWD


def main():
    digits = load_digits()
    keep = (digits.target == 7) | (digits.target == 9)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data[keep], digits.target[keep],
        test_size=0.25, stratify=digits.target[keep], random_state=42,
    )
    model = Pipeline([
        ('scale', StandardScaler()),
        ('ensemble', EnsembleKernelDWD()),
    ])
    model.fit(X_train, y_train)
    fitted = model.named_steps['ensemble']
    print(f'Test accuracy: {model.score(X_test, y_test):.4f}')
    print(f'Classes: {fitted.classes_.tolist()}')
    print(f'Base learners: {len(fitted.estimators_)}')
    print(f'Selected training rows: {len(fitted.selected_indices_)}')
    print(f'Shared gamma: {fitted.gamma_}')
    print(f'Iterations, base learners then final: {fitted.n_iter_}')


if __name__ == '__main__':
    main()
