// Mirrors the FastAPI responses in backend/api/*.py.

export type ClassName = "Pure Fresh" | "Good" | "Stale to Spoiled";
export type Features = Record<string, number>;

export interface Health {
  status: "ok" | "degraded";
  xai_model?: string;
  detail?: string;
}

export interface FeatureStat {
  mean: number;
  std: number;
  min: number;
  median: number;
  max: number;
}

export interface Dataset {
  samples: number;
  features: number;
  classes: ClassName[];
  feature_names: string[];
  class_distribution: Record<string, number>;
  raw_class_distribution: Record<string, number>;
  missing_values: number;
  train_size: number;
  test_size: number;
  feature_stats: Record<string, FeatureStat>;
}

export interface SampleRef {
  index: number;
  actual_class: ClassName;
}

export interface Sample extends SampleRef {
  features: Features;
}

export interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  f1_macro: number;
  cv_accuracy_mean: number;
  cv_accuracy_std: number;
  cv_f1_mean: number;
  cv_f1_std: number;
  confusion_matrix: number[][];
  selected_for_xai: boolean;
}

export interface ModelSelection {
  xai_selected_model: string;
  selection_metric: string;
  selection_value: number;
  tie_breaker_metric: string;
  tie_breaker_value: number;
  selection_data: string;
  test_set_used_for_selection: boolean;
}

export interface ModelsResponse {
  class_order: ClassName[];
  selection: ModelSelection;
  models: ModelMetrics[];
}

export interface Prediction {
  predicted_class: ClassName;
  confidence: number;
  probabilities: Record<ClassName, number>;
  model: string;
}

export interface FeatureImportance {
  model: string;
  method: string;
  explanation_type: "global";
  features: { name: string; importance: number }[];
}

interface ExplainBase {
  model: string;
  explanation_type: "local";
  prediction: ClassName;
  confidence: number;
  probabilities: Record<ClassName, number>;
  sample_index: number | null;
  actual_class: ClassName | null;
}

export interface LimeExplanation extends ExplainBase {
  method: "LIME";
  explained_class: ClassName;
  intercept: number;
  contributions: { feature: string; rule: string; value: number; contribution: number }[];
}

export interface ShapLocal extends ExplainBase {
  method: "SHAP";
  feature_names: string[];
  feature_values: number[];
  class_names: ClassName[];
  base_values: Record<ClassName, number>;
  shap_values: Record<ClassName, number[]>;
  output_values: Record<ClassName, number>;
}

export interface ShapGlobal {
  model: string;
  explanation_type: "global";
  explained_class: ClassName;
  n_samples: number;
  importance: { feature: string; mean_abs_shap: number; mean_abs_shap_class: number }[];
  summary: {
    feature: string;
    feature_min: number;
    feature_max: number;
    points: { shap: number; value: number }[];
  }[];
}

export type ExplainInput = { sample_index: number } | { features: Features };
