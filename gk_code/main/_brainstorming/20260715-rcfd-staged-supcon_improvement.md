# Regression-Conditioned Classification via Multi-Stage SupCon

## Core Philosophy
This architecture solves the "Continuous Domain Shift" problem in time-series data (e.g., PCR concentration delays). Instead of forcing a single classification backbone to untangle physical shifts from biological targets simultaneously, we use a modular, physics-first approach: **Unwarp the physics, then identify the biology.**

---

## The Oracle Experiment (Validation Phase)
Before engineering the regression model, we establish the theoretical performance ceiling by feeding ground truth target variables directly into the conditioning layer.

### The Success Criteria
Compare the Oracle against a standard Baseline model (Dual Backbone $\to$ Cross-Entropy). 
*   **Green Light (Significant Gap):** e.g., 85% $\to$ 95%. The conditioning works perfectly. Proceed to Phase 1.
*   **Red Light (No Gap / Worse):** e.g., 85% $\to$ 84%. The classifier is ignoring the conditioning layer or gradients are collapsing. Debug architecture.
*   **Abandon Ship (Marginal Gap):** e.g., 85% $\to$ 86%. The theoretical ceiling is too low to survive the noise introduced by a real, imperfect regression model.

---

## 3-Phase Training Pipeline

If the Oracle Experiment passes, execute the following serialized training pipeline to prevent gradient collision between the regression, clustering, and classification goals.

### Phase 1: Train the Regression Oracle
*   **Objective:** Build a robust, offline model dedicated entirely to predicting the physical confounder (concentration).
*   **Architecture:** Deeper/Wider 1D-CNN or ResNet capable of mapping raw curve shape to a scalar prediction. 
*   **Loss:** Mean Squared Error (MSE) or Mean Absolute Error (MAE).
*   **Action:** Train to convergence. Save the model to disk (`best_regressor.keras`). **Freeze these weights permanently.**

### Phase 2: Conditioned Supervised Contrastive Learning (SupCon)
*   **Objective:** Force the dual backbone to build a tight, domain-invariant latent space where targets are perfectly clustered regardless of their original concentration.
*   **Data Flow:** 
    1. Pass raw curve to the Frozen Regression Oracle $\to$ get scalar `c_pred`.
    2. Pass raw curve to Dual Backbone $\to$ get raw embeddings.
    3. Apply FiLM conditioning: modulate raw embeddings using `c_pred` to mathematically unwarp the physical delay.
*   **Loss:** SupCon Loss applied to the conditioned embeddings.
*   **Action:** Train until the Silhouette Score plateaus. **Freeze the Dual Backbone and FiLM layers.**

### Phase 3: Linear Evaluation (Classification)
*   **Objective:** Draw optimal decision boundaries through the cleanly separated latent clusters.
*   **Architecture:** Attach a final Dense layer (the classification head) to the frozen conditioned embeddings.
*   **Loss:** Cross-Entropy.
*   **Action:** Train only the classification head. Because the latent space is already perfectly arranged and concentration-corrected, this phase is fast and highly stable.