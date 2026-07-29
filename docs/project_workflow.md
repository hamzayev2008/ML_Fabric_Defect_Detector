                 START
                   │
                   ▼
        Collect Dataset
                   │
                   ▼
          Label Images
                   │
                   ▼
       Data Preprocessing
                   │
                   ▼
      Data Augmentation
                   │
                   ▼
  Train / Validation / Test Split
                   │
                   ▼
        Model Development
                   │
                   ▼
          Model Training
                   │
                   ▼
         Model Evaluation
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   Performance OK?        No
          │                 │
         Yes                │
          │                 │
          ▼                 │
   Save Best Model ◄────────┘
          │
          ▼
   Test on New Images
          │
          ▼
      Final Prediction
          │
          ▼
          END
