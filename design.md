# Tomato Freshness AI — Frontend Design Specification

## 1. Purpose

Implement the frontend for the **Tomato Freshness Classification + Explainable AI** project using **Next.js, React, TypeScript, and Tailwind CSS**.

The attached reference image is the visual direction for the application.

The reference is a clean, rounded, modern analytics dashboard with:

- A fixed left sidebar
- Large rounded content panels
- A light/off-white canvas
- White cards
- Dark green primary accents
- Large typography
- Generous whitespace
- Soft borders and shadows
- Compact analytics cards
- Data visualization panels
- A right-side contextual information column
- Subtle hover and transition effects

### Critical design rule

The reference image is being used for **visual language and layout**, not for its business content.

Do **not** copy the task-management content from the reference.

Replace it with content relevant to:

> **Tomato Freshness Classification & Explainable AI**

The ML functionality, page content, data, metrics, model names, LIME/SHAP outputs, and terminology must come from the actual application/backend.

---

# 2. Technology

Use:

```text
Next.js
React
TypeScript
Tailwind CSS
shadcn/ui where useful
Lucide React
Recharts
```

Use the Next.js **App Router**.

Do not introduce unnecessary frontend frameworks.

---

# 3. Visual Direction

## Overall Style

The UI should feel like:

```text
Modern ML analytics dashboard
+
Scientific data visualization
+
Premium SaaS dashboard
```

It should NOT feel like:

```text
Generic admin panel
Old Bootstrap dashboard
Generic AI chatbot
University project template
```

The interface should look polished enough to present during a project demonstration.

---

# 4. Color System

The reference uses a green-on-white visual identity.

Use a restrained green palette.

### Primary

```text
Primary dark green:
#064E3B

Primary green:
#047857

Accent green:
#10B981

Soft green:
#D1FAE5
```

### Background

```text
Application background:
#F5F7F5

Card background:
#FFFFFF

Subtle background:
#F8FAF9
```

### Text

```text
Primary text:
#111827

Secondary text:
#6B7280

Muted text:
#9CA3AF
```

### Borders

```text
#E5E7EB
```

### Status colors

Use sparingly:

```text
Success:
#16A34A

Warning:
#D97706

Error:
#DC2626

Info:
#2563EB
```

Do not turn the application into a rainbow dashboard.

Green should remain the dominant visual identity.

---

# 5. Typography

Use a modern sans-serif font.

Recommended:

```text
Inter
```

or an equivalent clean system sans-serif.

Hierarchy:

```text
Page title:
32–40px
font-weight: 600–700

Section heading:
20–24px
font-weight: 600

Card heading:
16–18px
font-weight: 600

Body:
14–16px

Secondary:
12–14px
```

The reference uses large, confident headings with substantial whitespace.

Do not overcrowd text.

---

# 6. Global Layout

Desktop layout:

```text
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│ ┌───────────────┐ ┌─────────────────────────────────────────────┐ │
│ │               │ │ Top Header                                 │ │
│ │               │ ├─────────────────────────────────────────────┤ │
│ │   Sidebar     │ │                                             │ │
│ │               │ │              Main Content                    │ │
│ │               │ │                                             │ │
│ │               │ │                                             │ │
│ │               │ │                                             │ │
│ │               │ │                                             │ │
│ └───────────────┘ └─────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Sidebar

Approximately:

```text
240–260px
```

### Main content

Flexible width.

### Outer spacing

Use approximately:

```text
24–32px
```

### Card gaps

Use:

```text
16–20px
```

---

# 7. Sidebar

The sidebar should closely follow the reference image.

Structure:

```text
┌───────────────────────┐
│  ◉ TomatoAI           │
│                       │
│  ANALYTICS            │
│                       │
│  ▦ Overview            │
│  ◉ Prediction          │
│  ▥ Models              │
│  ◇ Features            │
│  ✦ LIME                │
│  ✦ SHAP                │
│  ▤ Dataset             │
│                       │
│  SYSTEM               │
│                       │
│  ⚙ Settings            │
│  ? About               │
│                       │
│                       │
│  ┌─────────────────┐  │
│  │ ML Pipeline     │  │
│  │ ● Connected     │  │
│  └─────────────────┘  │
└───────────────────────┘
```

Use Lucide icons.

Do not use emoji icons in the actual UI.

---

# 8. Sidebar Interaction

Inactive navigation item:

```text
background: transparent
text: muted gray
```

Hover:

```text
background: #F0FDF4
text: primary green
```

Active item:

```text
background: #E8F5EF
text: #064E3B
font-weight: 600
```

Add a thin green indicator on the left edge of the active item, similar to the reference.

Transition:

```text
150–200ms ease
```

---

# 9. Header

The header should visually resemble the reference.

Left:

```text
Search
```

Right:

```text
Backend status
Notifications
User/project identity
```

However, do not create unnecessary fake user functionality.

Suggested:

```text
[ Search models, features, samples... ]     [● API Connected] [◎]
```

The search field does not need to implement full global search unless required.

It can initially be visual/UI-only.

---

# 10. Dashboard Page

Route:

```text
/
```

This is the primary page.

The dashboard should adapt the reference's card-based composition to ML.

## Top Section

```text
Tomato Freshness AI

Analyze tomato freshness with machine learning
and explain every prediction.

[ Run Prediction ]   [ View Models ]
```

Buttons:

```text
Run Prediction
```

Primary filled green.

```text
View Models
```

White outlined button.

---

# 11. Dashboard Metric Cards

Replace the reference's project counters with ML metrics.

Use four cards:

```text
┌────────────────────┐
│ Dataset Samples    │
│                    │
│ 555                │
│                    │
│ 26 features       │
└────────────────────┘

┌────────────────────┐
│ Freshness Classes  │
│                    │
│ 3                  │
│                    │
│ Pure Fresh / Good  │
└────────────────────┘

┌────────────────────┐
│ Best CV F1         │
│                    │
│ Actual value       │
│                    │
│ 5-fold weighted F1 │
└────────────────────┘

┌────────────────────┐
│ Test Accuracy      │
│                    │
│ Actual value       │
│                    │
│ Selected model     │
└────────────────────┘
```

### Important

Never hard-code metric values.

Load them from the backend artifacts/API.

---

# 12. Primary Metric Card

Following the reference's prominent green card, the first or most important metric can use a dark-green background.

Example:

```text
Selected Model

Random Forest

CV Weighted F1
98.89%

5-fold stratified CV
```

Use:

```text
dark green background
white text
```

with a small trend/status indicator.

Do not display fake "increased from last month" language.

Use meaningful ML metadata instead.

---

# 13. Model Performance Panel

Large card:

```text
┌─────────────────────────────────────────────┐
│ Model Performance                           │
│                                             │
│ Accuracy                                    │
│                                             │
│ RF      ████████████████████                │
│ SVM     █████████████████                   │
│ XGB     ████████████████████                │
│                                             │
│ [Accuracy] [Precision] [Recall] [F1]        │
└─────────────────────────────────────────────┘
```

Use Recharts.

Allow metric switching:

```text
Accuracy
Precision
Recall
F1
```

Do not use 3D charts.

Do not use excessive gradients.

---

# 14. Freshness Distribution Card

Create a card for class distribution.

```text
Freshness Distribution

Pure Fresh          █████████████████
Good                ███████
Stale to Spoiled    ████
```

Use a clean bar chart.

Display actual counts.

The chart should make the class imbalance immediately visible.

---

# 15. XAI Overview Card

Create a contextual card similar to the reference's "Reminders" panel.

Title:

```text
Explainable AI
```

Content:

```text
Selected model
Random Forest

Global explanation
SHAP

Local explanation
LIME

Latest explained sample
#6
```

Actions:

```text
[View LIME]
[View SHAP]
```

---

# 16. Feature Importance Card

Create a medium/large card:

```text
Top Predictive Features

MQ135_mean                 ████████████
MQ138_mean                 ██████████
MQ136_over_MQ138_end       ████████
MQ136_mean                 ███████
MQ3_mean                   ██████
```

Show the top 5 features.

Add:

```text
View all features →
```

which routes to:

```text
/features
```

---

# 17. Recent Prediction / Sample Card

Instead of "Project" from the reference, show:

```text
Recent Prediction
```

Example structure:

```text
Sample #6

Predicted:
Stale to Spoiled

Actual:
Stale to Spoiled

Confidence:
100%

[Explain]
```

Do not invent a prediction.

If there is no prediction yet:

```text
No predictions yet.

Run your first prediction →
```

---

# 18. Prediction Page

Route:

```text
/predict
```

This is the interactive ML page.

Use a large two-column layout:

```text
┌───────────────────────────────────────┬───────────────────────┐
│ Sensor Inputs                         │ Prediction             │
│                                       │                       │
│ MQ2                                    │ Pure Fresh             │
│ MQ3                                    │                       │
│ MQ9                                    │ Confidence             │
│ MQ135                                  │ 94.2%                 │
│ MQ136                                  │                       │
│ MQ138                                  │ Probabilities          │
│ Temperature                            │                       │
│ Humidity                               │                       │
│ Ratios                                 │                       │
│                                       │                       │
│ [Predict Freshness]                   │ [Explain]              │
└───────────────────────────────────────┴───────────────────────┘
```

---

# 19. Prediction Input Design

Group inputs by sensor.

Sections:

```text
MQ2
MQ3
MQ9
MQ135
MQ136
MQ138
Environment
Derived Ratios
```

Each feature should use:

```text
Label
Numeric input
Optional unit/helper text
```

Use a compact grid.

Example:

```text
MQ2 Mean        MQ2 Std        MQ2 Slope
[ 1.24 ]        [ 0.14 ]       [ 0.03 ]
```

Avoid a giant single-column form.

---

# 20. Prediction Result

Use a visually prominent result card.

```text
┌─────────────────────────────────┐
│ Predicted Freshness             │
│                                 │
│     STALE TO SPOILED            │
│                                 │
│ Confidence                      │
│     96.8%                       │
│                                 │
│ Model                           │
│ Random Forest                   │
└─────────────────────────────────┘
```

Color should communicate state subtly.

Do not use aggressive red unless appropriate.

---

# 21. Probability Visualization

Show all three class probabilities.

```text
Pure Fresh       4.2%
Good             11.3%
Stale to Spoiled 84.5%
```

Use horizontal bars.

Always display all classes.

---

# 22. Models Page

Route:

```text
/models
```

Use the reference's card/table style.

Header:

```text
Model Performance
Compare the three supervised classifiers.
```

Display:

```text
Random Forest
SVM
XGBoost
```

Metrics:

```text
CV F1
Accuracy
Precision
Recall
Test F1
```

Use actual backend data.

---

# 23. Model Cards

Each model can have a compact card:

```text
Random Forest

CV Weighted F1
98.89%

Test Accuracy
99.10%

Status
Selected for XAI
```

The selected model gets the primary green treatment.

Do not call it "best model" based only on test accuracy.

Use:

```text
Selected for XAI
```

because selection is based on CV performance.

---

# 24. Confusion Matrix Section

Display three confusion matrices.

Desktop:

```text
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Random Forest   │ │ SVM             │ │ XGBoost         │
│                 │ │                 │ │                 │
│   matrix        │ │   matrix        │ │   matrix        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

On smaller screens stack them vertically.

Use consistent class ordering:

```text
Pure Fresh
Good
Stale to Spoiled
```

---

# 25. Feature Importance Page

Route:

```text
/features
```

Main visual:

```text
Feature Importance

MQ135_mean                 ███████████████
MQ138_mean                 █████████████
MQ136_over_MQ138_end       ███████████
...
```

Use horizontal bars because feature names are long.

Add a side panel:

```text
How to interpret this

Higher importance means the feature
contributes more to the model's
predictions.

Importance does not imply causation.
```

---

# 26. LIME Page

Route:

```text
/explain/lime
```

Use a focused analytical layout.

Top:

```text
LIME — Local Explanation

Understand why the model made one
specific prediction.
```

Sample selector:

```text
Sample
[ #6 ▼ ]
```

Prediction summary:

```text
Actual
Stale to Spoiled

Predicted
Stale to Spoiled

Confidence
100%
```

Main visualization:

```text
Features supporting prediction

MQ138_mean       ███████████
Humidity_mean    █████████
MQ136_mean       ███████

Features opposing prediction

MQ2_std          ───
...
```

Include an explanation card:

```text
LIME is a local explanation method.
It approximates the model around the
selected sample.
```

---

# 27. SHAP Page

Route:

```text
/explain/shap
```

Use three sections.

## Section 1 — Global Importance

```text
SHAP Feature Importance
```

Bar chart.

## Section 2 — SHAP Summary

Use a beeswarm-like visualization if available.

## Section 3 — Local Explanation

For the selected sample:

```text
Why was this sample classified this way?
```

Display SHAP contribution values.

---

# 28. LIME vs SHAP Comparison

Because this is an Explainable AI academic project, add a small comparison card.

```text
LIME vs SHAP

LIME
Local approximation
Fast and model-agnostic

SHAP
Shapley-value based
Global + local interpretation
```

Do not claim one explanation method is universally superior.

---

# 29. Dataset Page

Route:

```text
/dataset
```

Show:

```text
555 Samples
26 Features
3 Target Classes
0 Missing Values
```

Then:

```text
Class Distribution
```

and:

```text
Feature List
```

Use expandable groups:

```text
MQ2
MQ3
MQ9
MQ135
MQ136
MQ138
Environment
Ratios
```

---

# 30. Settings/About

Keep these minimal.

Settings can contain:

```text
API status
Backend URL
Model artifact version
Dataset version
```

Do not implement user accounts or authentication unless explicitly required.

---

# 31. Card Design

All cards should use:

```text
background: white
border: 1px solid #E5E7EB
border-radius: 20–24px
```

Soft shadow:

```text
0 4px 20px rgba(0,0,0,0.04)
```

Do not make shadows heavy.

The reference relies primarily on:

```text
rounded corners
spacing
subtle borders
color blocks
```

rather than dramatic shadows.

---

# 32. Border Radius

Use consistently:

```text
Small controls:
10–12px

Cards:
18–22px

Large panels:
22–28px

Pills:
9999px
```

Do not mix many unrelated radii.

---

# 33. Buttons

Primary:

```text
dark green background
white text
rounded-full or 12px radius
```

Secondary:

```text
white background
green/dark border
dark text
```

Hover:

```text
slightly darker green
small upward/scale transition
```

Example:

```text
[ Run Prediction ]
```

should feel tactile.

---

# 34. Hover Effects

Keep effects subtle.

Cards:

```text
translateY(-2px)
shadow slightly increases
border becomes slightly darker
```

Navigation:

```text
background fades in
icon/text shifts toward green
```

Buttons:

```text
scale(1.01)
brightness change
```

Charts:

Use native tooltip interactions.

Do not make every element bounce or glow.

---

# 35. Page Transitions

Use subtle transitions between routes.

Preferred:

```text
opacity: 0 → 1
translateY: 4px → 0
duration: 200–300ms
```

Do not use large cinematic page transitions.

This is an analytics dashboard.

---

# 36. Scroll Behavior

Main page:

```text
smooth scrolling where appropriate
```

Sidebar:

```text
sticky/fixed
```

Main content:

```text
scroll independently if necessary
```

On long explanation pages, section navigation may be sticky.

---

# 37. Chart Animation

Charts should animate when first rendered.

Use short animations:

```text
400–700ms
```

Avoid excessive looping animations.

Do not animate every metric continuously.

---

# 38. Loading States

Every API-driven component needs a loading state.

Use skeleton cards:

```text
████████████████
████████
```

instead of blank areas.

Prediction:

```text
Analyzing sample...
```

LIME:

```text
Generating local explanation...
```

SHAP:

```text
Computing SHAP values...
```

---

# 39. Error States

If the backend is unavailable:

```text
ML service unavailable

The frontend could not connect to
the FastAPI backend.

[ Retry ]
```

Do not expose stack traces.

For prediction errors:

```text
Unable to generate prediction.
Check the input values and try again.
```

---

# 40. Empty States

If no predictions exist:

```text
No predictions yet

Enter sensor values to generate
your first freshness prediction.

[ Run Prediction ]
```

If no explanation exists:

```text
No explanation available

Run a prediction first.
```

---

# 41. Responsive Design

The desktop reference is the primary design target.

Also support:

### Tablet

Sidebar collapses.

### Mobile

Layout:

```text
Top header
     ↓
Content
     ↓
Bottom navigation / menu
```

Cards become single-column.

Charts remain horizontally scrollable where necessary.

The prediction form should become:

```text
single-column
```

on small screens.

---

# 42. Accessibility

Follow basic accessibility requirements:

- Semantic HTML
- Keyboard navigable controls
- Visible focus states
- Proper labels for all inputs
- ARIA labels where needed
- Sufficient color contrast
- Do not communicate information by color alone
- Charts should have textual summaries where practical

---

# 43. Frontend Data Rules

The frontend must never fabricate ML values.

Do not hard-code:

```text
Accuracy
F1
Feature importance
SHAP values
LIME values
Prediction probabilities
```

except as temporary development placeholders that are removed before completion.

All final values come from FastAPI.

Expected flow:

```text
Next.js
   ↓
FastAPI
   ↓
Saved ML artifacts
   ↓
JSON
   ↓
React components
```

---

# 44. API Integration

Use a centralized API client:

```text
frontend/lib/api.ts
```

Functions:

```text
getHealth()
getDataset()
getModels()
getFeatureImportance()
predict(features)
explainLime(features)
explainShap(features)
```

Keep API communication out of individual low-level chart components.

---

# 45. Type Safety

Define TypeScript interfaces in:

```text
frontend/lib/types.ts
```

Examples:

```ts
interface ModelMetrics {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  cvF1Mean?: number
  cvF1Std?: number
}

interface PredictionResult {
  predictedClass: string
  probabilities: Record<string, number>
  model: string
}
```

Avoid:

```ts
any
```

unless absolutely necessary.

---

# 46. Component Architecture

Use reusable components.

Recommended:

```text
components/
├── layout/
│   ├── Sidebar.tsx
│   ├── Header.tsx
│   └── PageContainer.tsx
│
├── dashboard/
│   ├── MetricCard.tsx
│   ├── ModelPerformance.tsx
│   ├── FreshnessDistribution.tsx
│   ├── XAIOverview.tsx
│   └── FeatureImportancePreview.tsx
│
├── prediction/
│   ├── SensorInputForm.tsx
│   ├── PredictionCard.tsx
│   └── ProbabilityChart.tsx
│
├── models/
│   ├── ModelCard.tsx
│   ├── MetricsTable.tsx
│   └── ConfusionMatrix.tsx
│
├── xai/
│   ├── LimeExplanation.tsx
│   ├── ShapSummary.tsx
│   ├── ShapLocalExplanation.tsx
│   └── XaiComparison.tsx
│
└── charts/
    ├── MetricChart.tsx
    ├── DistributionChart.tsx
    └── FeatureImportanceChart.tsx
```

---

# 47. Animation Philosophy

Use animation to communicate state, not decoration.

Good:

```text
Card entrance
Chart entrance
Button hover
Sidebar hover
Prediction result reveal
Loading skeleton
```

Avoid:

```text
constant floating elements
large parallax effects
excessive glowing
continuous chart movement
```

The reference is calm and professional.

Keep that feeling.

---

# 48. Visual Relationship Between ML and UI

The application should make the ML workflow visually understandable.

Use the following hierarchy:

```text
DATASET
   ↓
MODELS
   ↓
PREDICTION
   ↓
EXPLANATION
```

The navigation should mirror this mental model.

The user should understand the project within a few seconds of opening the dashboard.

---

# 49. Important Academic Presentation Rules

This is an academic ML/XAI project.

The UI must not make unsupported scientific claims.

Avoid language such as:

```text
"Perfect model"
"100% reliable"
"Guaranteed freshness"
"Proves spoilage"
```

Instead:

```text
"Model prediction"
"Confidence"
"Predictive feature importance"
"Model explanation"
```

Feature importance must not be described as causation.

LIME and SHAP explanations must be described as explanations of model behavior.

---

# 50. Reference Image Mapping

Use the uploaded dashboard image as follows:

### Reference → Tomato AI

```text
Donezo logo
        ↓
TomatoAI logo/name

Dashboard
        ↓
Overview

Total Projects
        ↓
Dataset Samples

Ended Projects
        ↓
Freshness Classes

Running Projects
        ↓
CV Performance

Pending Project
        ↓
Test Accuracy

Project Analytics
        ↓
Model Performance

Reminders
        ↓
Explainable AI

Project List
        ↓
Recent Predictions

Team Collaboration
        ↓
Feature Importance / Model Insights

Project Progress
        ↓
Freshness Distribution / Prediction Confidence

Time Tracker
        ↓
Model/XAI Status
```

The **layout and visual hierarchy** should resemble the reference, while the content should be entirely Tomato Freshness AI-specific.

---

# 51. Do Not Copy

Do not copy these from the reference:

- Donezo branding
- Task/project terminology
- User names
- Project names
- Calendar content
- Meeting content
- Fake due dates
- Time tracker
- Mobile app advertisement
- Avatars
- Fake analytics
- Fake metrics

Only use the reference for:

```text
Layout
Spacing
Card composition
Typography hierarchy
Green/white palette
Rounded geometry
Dashboard density
Interaction feel
```

---

# 52. Suggested Dashboard Composition

The final desktop dashboard should roughly follow this composition:

```text
┌──────────────┬─────────────────────────────────────────────────────┐
│              │ Header / Search / Status                           │
│              ├─────────────────────────────────────────────────────┤
│              │                                                     │
│   Sidebar    │ Title                         Actions              │
│              │                                                     │
│              ├──────────┬──────────┬──────────┬──────────────────┤
│              │ Dataset  │ Classes  │ CV F1    │ Test Accuracy    │
│              └──────────┴──────────┴──────────┴──────────────────┘
│              │                                                     │
│              ├─────────────────────────┬───────────────────────────┤
│              │ Model Performance        │ Explainable AI            │
│              │                         │                           │
│              │ Chart                   │ LIME / SHAP               │
│              │                         │                           │
│              ├─────────────────────────┼───────────────────────────┤
│              │ Freshness Distribution  │ Recent Prediction         │
│              │                         │                           │
│              │ Chart                   │ Result + Confidence      │
│              ├─────────────────────────┴───────────────────────────┤
│              │ Top Feature Importance                              │
│              │                                                     │
└──────────────┴─────────────────────────────────────────────────────┘
```

---

# 53. Implementation Priority

Build in this order.

## Phase 1 — Layout

1. App shell
2. Sidebar
3. Header
4. Main content container
5. Responsive behavior

## Phase 2 — Dashboard

1. Metric cards
2. Model performance chart
3. Freshness distribution
4. Feature importance
5. XAI overview
6. Recent prediction

## Phase 3 — Prediction

1. Sensor form
2. API integration
3. Prediction result
4. Probability chart

## Phase 4 — ML Analysis

1. Models page
2. Confusion matrices
3. Feature importance page
4. Dataset page

## Phase 5 — XAI

1. LIME page
2. SHAP page
3. LIME vs SHAP comparison

## Phase 6 — Polish

1. Hover effects
2. Page transitions
3. Loading states
4. Error states
5. Empty states
6. Responsive fixes
7. Accessibility
8. Final visual refinement

---

# 54. Performance Requirements

Do not perform expensive ML computation in React components.

The frontend should call:

```text
FastAPI
```

for ML operations.

Do not:

- load Python models into the browser
- run SHAP in the browser
- run LIME in the browser
- duplicate preprocessing in TypeScript

The browser is responsible only for presentation and interaction.

---

# 55. Definition of Done

The frontend is complete when:

- [ ] Next.js App Router is implemented.
- [ ] Sidebar matches the reference visual language.
- [ ] Header matches the reference visual language.
- [ ] Dashboard uses rounded white cards.
- [ ] Green/white visual identity is consistent.
- [ ] Dashboard displays real dataset statistics.
- [ ] Dashboard displays real model metrics.
- [ ] Dashboard displays real feature importance.
- [ ] Model comparison page works.
- [ ] Confusion matrices are displayed.
- [ ] Prediction form accepts all 26 features.
- [ ] Prediction calls FastAPI.
- [ ] Prediction probabilities are displayed.
- [ ] LIME page works.
- [ ] SHAP page works.
- [ ] Dataset page works.
- [ ] Loading states exist.
- [ ] Error states exist.
- [ ] Empty states exist.
- [ ] Responsive layout works.
- [ ] Keyboard navigation works.
- [ ] No final ML values are hard-coded.
- [ ] No fake business/project data remains.
- [ ] UI uses actual backend results.
- [ ] Hover interactions are subtle.
- [ ] Animations are subtle.
- [ ] The final application visually follows the uploaded reference without copying its content.

---

# 56. Final Design Principle

The application should feel like a **premium analytical instrument for understanding tomato freshness predictions**.

The visual hierarchy should communicate:

```text
What is the dataset?
        ↓
How well do the models perform?
        ↓
What does the model predict?
        ↓
Why did it make that prediction?
```

The reference image provides the visual language.

The actual Tomato Freshness ML/XAI system provides the content.
