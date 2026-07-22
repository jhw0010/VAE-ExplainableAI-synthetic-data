import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import List, Optional

class TVAEInterpreter:
    """Computes First-Order (Feature Importance) and Second-Order (Interaction) Gradients."""
    
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def compute_first_order_gradients(self, dataset: tf.data.Dataset):
        """Calculates gradients of mean and logvar with respect to inputs."""
        mus, logvars = [], []
        mu_grads, logvars_grads = [], []

        for train_x in dataset:
            with tf.GradientTape(persistent=True) as g:
                g.watch(train_x)
                mean, logvar = self.model.encode(train_x)
                mus.append(mean)
                logvars.append(logvar)

                mu_i_grads = [g.gradient(mean[:, i], train_x) for i in range(mean.shape[1])]
                logvars_i_grads = [g.gradient(logvar[:, i], train_x) for i in range(logvar.shape[1])]

                mu_grads.append(mu_i_grads)
                logvars_grads.append(logvars_i_grads)

        return mu_grads, logvars_grads

    def compute_global_importance(self, mu_grads: List, logvars_grads: List) -> np.ndarray:
        """Aggregates first-order gradients to obtain global relative feature importance."""
        sij_mu_xn_square, sij_sigma_xn_square = [], []

        for j in range(len(mu_grads[0])):
            sij_mu_j = [mu_grads[i][j] ** 2 for i in range(len(mu_grads))]
            sij_sigma_j = [logvars_grads[i][j] ** 2 for i in range(len(logvars_grads))]
            sij_mu_xn_square.append(sij_mu_j)
            sij_sigma_xn_square.append(sij_sigma_j)

        sij_mu_sq, sij_sigma_sq = [], []
        for j in range(len(sij_mu_xn_square)):
            tmp_mu = tf.concat(sij_mu_xn_square[j], axis=0)
            tmp_logvar = tf.concat(sij_sigma_xn_square[j], axis=0)
            sij_mu_sq.append(tf.math.sqrt(tf.reduce_mean(tmp_mu, axis=0, keepdims=True)))
            sij_sigma_sq.append(tf.math.sqrt(tf.reduce_mean(tmp_logvar, axis=0, keepdims=True)))

        sj_mu_sq = tf.reduce_sum(tf.concat(sij_mu_sq, axis=0), axis=0, keepdims=True)
        sj_sigma_sq = tf.reduce_sum(tf.concat(sij_sigma_sq, axis=0), axis=0, keepdims=True)
        
        sj_sq = (sj_mu_sq + sj_sigma_sq).numpy().reshape(-1)
        relative_importance = (sj_sq / sj_sq.sum()) * 100
        return relative_importance

    def compute_second_order_interactions(self, train_x: tf.Tensor) -> tf.Tensor:
        """Computes second-order batch Jacobians to analyze inter-feature dependencies."""
        with tf.GradientTape(persistent=True) as g:
            with tf.GradientTape(persistent=True) as g2:
                g.watch(train_x)
                g2.watch(train_x)
                mean, logvar = self.model.encode(train_x)

                mjo_xn = []
                accumulated = 0
                for i in range(mean.shape[1]):
                    dmu_i = g.gradient(mean[:, i], train_x)
                    dsigma_i = g.gradient(logvar[:, i], train_x)

                    accumulated += tf.abs(g2.batch_jacobian(dmu_i, train_x))
                    accumulated += tf.abs(g2.batch_jacobian(dsigma_i, train_x))
                    mjo_xn.append(accumulated)

        m_jj_xn = tf.add_n(mjo_xn)
        mask = 1.0 - tf.eye(m_jj_xn.shape[1])
        m_j_xn = tf.reduce_sum(m_jj_xn * tf.expand_dims(mask, axis=0), axis=2)
        m_j = tf.reduce_mean(m_j_xn, axis=0)
        return m_j

    @staticmethod
    def plot_importance(importance_scores: np.ndarray, feature_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Generates publication-quality global importance visualization."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        indices = range(len(importance_scores))
        labels = feature_names if feature_names else [f"Feature {i+1}" for i in indices]

        bars = ax.bar(indices, importance_scores, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.set_title("Relative Global Feature Importance (%)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Features", fontsize=12, labelpad=10)
        ax.set_ylabel("Importance Score (%)", fontsize=12, labelpad=10)
        ax.set_xticks(indices)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
