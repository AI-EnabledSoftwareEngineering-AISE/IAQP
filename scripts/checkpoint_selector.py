#!/usr/bin/env python3
"""
Checkpoint Selector
==================
Intelligent checkpoint selection based on evaluation results.

Supports two selection modes:
1. "best": Selects checkpoint with best performance on its own backend
2. "generalization": Selects checkpoint with best cross-backend generalization

For exact_k backend, only "generalization" mode is valid since there's no ANN backend.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


class CheckpointSelector:
    """
    Intelligent checkpoint selector with support for different selection strategies.
    """
    
    def __init__(self):
        """Initialize the checkpoint selector."""
        self.supported_backends = ["cuvs_cagra", "cagra", "cagra_only", "ivf", "ivf_only", "exact_k"]
        self.supported_modes = ["best", "generalization"]
    
    def select_best_checkpoint(self, 
                              cagra_results: Dict[str, List[Any]],
                              ivf_results: Dict[str, List[Any]],
                              backend: str,
                              mode: str = "best",
                              at_k: List[int] = None) -> Dict[int, str]:
        """
        Select the best checkpoint based on evaluation results.
        
        Args:
            cagra_results: CAGRA evaluation results {variant: [(budget, recall, qps), ...]}
            ivf_results: IVF evaluation results {variant: [(budget, recall, qps), ...]}
            backend: Training backend ("cuvs_cagra", "ivf", "exact_k")
            mode: Selection mode ("best" or "generalization")
            at_k: list of k values to consider (select per-k)
            
        Returns:
            Dict of k -> best checkpoint epoch name (e.g., {10: "ep2"})
        """
        at_k = at_k or [10]
        if backend not in self.supported_backends:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {self.supported_backends}")
        
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported mode: {mode}. Supported: {self.supported_modes}")
        
        # For exact_k backend, only generalization mode is valid
        if backend == "exact_k" and mode != "generalization":
            print(f"⚠️  Warning: exact_k backend only supports 'generalization' mode, switching...")
            mode = "generalization"
        
        # Map simplified backend names to internal names
        backend_map = {
            "cagra": "cuvs_cagra",
            "cagra_only": "cuvs_cagra", 
            "ivf": "ivf",
            "ivf_only": "ivf",
            "exact_k": "exact_k"
        }
        internal_backend = backend_map.get(backend, backend)
        
        if mode == "best":
            return self._select_best_self_backend(cagra_results, ivf_results, internal_backend, at_k)
        elif mode == "generalization":
            return self._select_best_generalization(cagra_results, ivf_results, internal_backend, at_k)
    
    def _select_best_self_backend(self, 
                                 cagra_results: Dict[str, List[Any]],
                                 ivf_results: Dict[str, List[Any]],
                                 backend: str,
                                 at_k: List[int]) -> Dict[int, str]:
        """
        Select checkpoint with best performance on its own backend.
        
        For cuvs_cagra: Use CAGRA results
        For ivf: Use IVF results
        """
        if backend == "cuvs_cagra":
            return self._analyze_cagra_results(cagra_results, at_k)
        elif backend == "ivf":
            return self._analyze_ivf_results(ivf_results, at_k)
        else:
            raise ValueError(f"Cannot use 'best' mode for backend: {backend}")
    
    def _select_best_generalization(
        self,
        cagra_results: Dict[str, List[Any]],
        ivf_results: Dict[str, List[Any]],
        backend: str,
        at_k: List[int],
    ) -> Dict[int, str]:
        """
        Select checkpoint with best cross-backend generalization.
        Priority:
        1. Good performance on OTHER backend (not training backend)
        2. Doesn't drop significantly compared to baseline
        3. Reasonable performance on training backend (self-backend constraint)
        """
        projected_variants = [k for k in cagra_results.keys() if k.startswith("proj_")]
        if not projected_variants:
            raise ValueError("No projected variants found in results")

        baseline_cagra = {k: self._compute_weighted_score(cagra_results["baseline"], k) for k in at_k}
        baseline_ivf = {k: self._compute_weighted_score(ivf_results["baseline"], k) for k in at_k}
        print(f"🔍 Baseline performance per k: CAGRA={baseline_cagra}, IVF={baseline_ivf}")

        if backend in ["cuvs_cagra", "cagra", "cagra_only"]:
            training_backend = "cagra"
            other_backend = "ivf"
            other_results = ivf_results
            self_backend_results = cagra_results
        elif backend in ["ivf", "ivf_only"]:
            training_backend = "ivf"
            other_backend = "cagra"
            other_results = cagra_results
            self_backend_results = ivf_results
        else:
            training_backend = "exact_k"
            other_backend = "cagra"
            other_results = cagra_results
            self_backend_results = None

        epoch_scores = {k: {} for k in at_k}
        epoch_other_scores = {k: {} for k in at_k}
        epoch_self_scores = {k: {} for k in at_k} if self_backend_results else None

        for variant in projected_variants:
            epoch = variant.replace("proj_", "")
            for k in at_k:
                # scores for both backends
                cagra_score = self._compute_weighted_score(cagra_results[variant], k)
                ivf_score = self._compute_weighted_score(ivf_results[variant], k)
                epoch_other_scores[k][epoch] = self._compute_weighted_score(other_results[variant], k)
                if self_backend_results:
                    epoch_self_scores[k][epoch] = self._compute_weighted_score(self_backend_results[variant], k)

                if training_backend == "exact_k":
                    generalization_score = cagra_score
                else:
                    other_score = epoch_other_scores[k][epoch]
                    self_score = epoch_self_scores[k][epoch] if self_backend_results else 0.0
                    generalization_score = 0.7 * other_score + 0.3 * self_score
                epoch_scores[k][epoch] = generalization_score

        best_per_k = {}
        for k in at_k:
            valid_epochs = {}
            for epoch, score in epoch_scores[k].items():
                other_score = epoch_other_scores[k][epoch]
                baseline_other = baseline_cagra[k] if other_backend == "cagra" else baseline_ivf[k]
                min_other_score = baseline_other * 0.85

                if self_backend_results and epoch in epoch_self_scores[k]:
                    self_score = epoch_self_scores[k][epoch]
                    best_self_score = max(epoch_self_scores[k].values()) if epoch_self_scores[k] else 0.0
                    min_self_score = best_self_score * 0.80
                    if other_score >= min_other_score and self_score >= min_self_score:
                        valid_epochs[epoch] = score
                        print(f"✅ k={k} {epoch}: other={other_score:.4f} (min={min_other_score:.4f}), self={self_score:.4f} (min={min_self_score:.4f})")
                    else:
                        print(f"❌ k={k} {epoch}: other={other_score:.4f} (min={min_other_score:.4f}), self={self_score:.4f} (min={min_self_score:.4f})")
                else:
                    if other_score >= min_other_score:
                        valid_epochs[epoch] = score
                        print(f"✅ k={k} {epoch}: other={other_score:.4f} (min={min_other_score:.4f})")
                    else:
                        print(f"❌ k={k} {epoch}: other={other_score:.4f} (min={min_other_score:.4f})")

            if not valid_epochs:
                print(f"⚠️  k={k}: No epochs meet constraints, using best other-backend performance")
                valid_epochs = {ep: epoch_other_scores[k][ep] for ep in epoch_scores[k].keys()}

            best_epoch = max(valid_epochs.items(), key=lambda x: x[1])[0]
            best_per_k[k] = best_epoch
            print(f"📊 k={k} Generalization scores: {epoch_scores[k]}")
            print(f"📊 k={k} Other backend scores ({other_backend}): {epoch_other_scores[k]}")
            if epoch_self_scores:
                print(f"📊 k={k} Self backend scores ({training_backend}): {epoch_self_scores[k]}")
            print(f"🏆 k={k} Best generalization: {best_epoch} (score: {epoch_scores[k][best_epoch]:.4f})")

        return best_per_k
    
    def _extract_budget_and_recall(self, entry: Any, k: int = None):
        """
        Extract budget/nprobe and recall value from a result entry.
        Supports legacy tuples (budget, recall, qps) and dict entries with per-k recall maps.
        """
        budget = None
        recall_val = None
        if isinstance(entry, dict):
            budget = entry.get("budget", entry.get("nprobe", entry.get("n_probe")))
            recall_data = entry.get("recall")
            if isinstance(recall_data, dict):
                if k is not None and k in recall_data:
                    recall_val = recall_data[k]
                elif k is not None and str(k) in recall_data:
                    recall_val = recall_data[str(k)]
                elif recall_data:
                    recall_val = next(iter(recall_data.values()))
            else:
                recall_val = recall_data
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            budget = entry[0]
            recall_val = entry[1]
        return budget, recall_val
    
    def _compute_weighted_score(self, results_data: List[Any], k: int = None) -> float:
        """
        Compute weighted score where higher budgets get more weight.
        
        Args:
            results_data: List of result entries (tuple or dict with recall map)
            
        Returns:
            Weighted average recall
        """
        if not results_data:
            return 0.0
        
        budgets = []
        recalls = []
        for entry in results_data:
            budget, recall_val = self._extract_budget_and_recall(entry, k)
            if budget is None or recall_val is None:
                continue
            budgets.append(int(budget))
            recalls.append(float(recall_val))
        
        if not budgets:
            return 0.0
        
        # Weight by budget (higher budgets get more weight)
        # Use budget as weight directly (budget 10 -> weight 10, budget 100 -> weight 100)
        weights = np.array(budgets, dtype=np.float32)
        recalls = np.array(recalls, dtype=np.float32)
        
        # Compute weighted average
        weighted_score = np.average(recalls, weights=weights)
        
        return float(weighted_score)
    
    def _analyze_cagra_results(self, cagra_results: Dict[str, List[Any]], at_k: List[int]) -> Dict[int, str]:
        """Analyze CAGRA results and select best checkpoint per k using weighted scoring."""
        projected_variants = [k for k in cagra_results.keys() if k.startswith("proj_")]
        
        if not projected_variants:
            raise ValueError("No projected variants found in CAGRA results")
        
        best_per_k = {}
        for k in at_k:
            epoch_scores = {}
            for variant in projected_variants:
                epoch = variant.replace("proj_", "")
                weighted_score = self._compute_weighted_score(cagra_results[variant], k)
                epoch_scores[epoch] = weighted_score
            best_epoch = max(epoch_scores.items(), key=lambda x: x[1])[0]
            best_per_k[k] = best_epoch
            print(f"📊 CAGRA k={k} scores (weighted by budget): {epoch_scores}")
            print(f"🏆 CAGRA k={k}: {best_epoch} (weighted recall: {epoch_scores[best_epoch]:.4f})")
        
        return best_per_k
    
    def _analyze_ivf_results(self, ivf_results: Dict[str, List[Any]], at_k: List[int]) -> Dict[int, str]:
        """Analyze IVF results and select best checkpoint per k using weighted scoring."""
        projected_variants = [k for k in ivf_results.keys() if k.startswith("proj_")]
        
        if not projected_variants:
            raise ValueError("No projected variants found in IVF results")
        
        best_per_k = {}
        for k in at_k:
            epoch_scores = {}
            for variant in projected_variants:
                epoch = variant.replace("proj_", "")
                weighted_score = self._compute_weighted_score(ivf_results[variant], k)
                epoch_scores[epoch] = weighted_score
            best_epoch = max(epoch_scores.items(), key=lambda x: x[1])[0]
            best_per_k[k] = best_epoch
            print(f"📊 IVF k={k} scores (weighted by budget): {epoch_scores}")
            print(f"🏆 IVF k={k}: {best_epoch} (weighted recall: {epoch_scores[best_epoch]:.4f})")
        
        return best_per_k
    
    def get_selection_summary(self, 
                            cagra_results: Dict[str, List[Any]],
                            ivf_results: Dict[str, List[Any]],
                            backend: str,
                            at_k: List[int] = None) -> Dict[str, Dict[int, str]]:
        """
        Get selection summary for both modes.
        
        Returns:
            Dictionary with selection results for both modes
        """
        summary = {}
        at_k = at_k or [10]
        
        # Map simplified backend names to internal names
        backend_map = {
            "cagra": "cuvs_cagra",
            "cagra_only": "cuvs_cagra", 
            "ivf": "ivf",
            "ivf_only": "ivf",
            "exact_k": "exact_k"
        }
        internal_backend = backend_map.get(backend, backend)
        
        if internal_backend == "exact_k":
            # Only generalization mode for exact_k
            summary["generalization"] = self.select_best_checkpoint(
                cagra_results, ivf_results, backend, "generalization", at_k
            )
        else:
            # Both modes for ANN backends
            summary["best"] = self.select_best_checkpoint(
                cagra_results, ivf_results, backend, "best", at_k
            )
            summary["generalization"] = self.select_best_checkpoint(
                cagra_results, ivf_results, backend, "generalization", at_k
            )
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Example results for testing
    example_cagra_results = {
        "baseline": [(10, 0.4, 1000), (20, 0.6, 2000)],
        "proj_ep1": [(10, 0.5, 1200), (20, 0.7, 2200)],
        "proj_ep2": [(10, 0.6, 1100), (20, 0.8, 2100)],
        "proj_ep3": [(10, 0.55, 1300), (20, 0.75, 2300)],
    }
    
    example_ivf_results = {
        "baseline": [(10, 0.3, 500), (20, 0.5, 1000)],
        "proj_ep1": [(10, 0.4, 600), (20, 0.6, 1100)],
        "proj_ep2": [(10, 0.5, 550), (20, 0.7, 1050)],
        "proj_ep3": [(10, 0.45, 650), (20, 0.65, 1150)],
    }
    
    # Test selector
    selector = CheckpointSelector()
    
    print("=== Testing Checkpoint Selector ===")
    
    # Test CAGRA backend
    print("\n--- CAGRA Backend ---")
    summary = selector.get_selection_summary(example_cagra_results, example_ivf_results, "cuvs_cagra")
    print(f"Selection summary: {summary}")
    
    # Test IVF backend
    print("\n--- IVF Backend ---")
    summary = selector.get_selection_summary(example_cagra_results, example_ivf_results, "ivf")
    print(f"Selection summary: {summary}")
    
    # Test exact_k backend
    print("\n--- Exact-K Backend ---")
    summary = selector.get_selection_summary(example_cagra_results, example_ivf_results, "exact_k")
    print(f"Selection summary: {summary}")
