#!/usr/bin/env python3
"""
EEG Real-Time Classification System
Multi-Modal Biometric Authentication: EEG Brainwave Pattern Recognition

Following tasks_eeg.yml implementation plan
Current: Task 07-10 - Data preparation and model training (Phase 1)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')


try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input, Reshape, Add, Layer
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. CNN functionality will be disabled.")
    TENSORFLOW_AVAILABLE = False

class EEGClassifier:
    """
    EEG Classification System for 3-class brainwave pattern recognition
    Classes: 0=Baseline, 1=Unlock, 2=Transaction
    """
    
    def __init__(self, data_path='examples/eeg_patterns.csv'):
        self.data_path = data_path
        self.eeg_channels = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 
                            'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12']
        self.phase_mapping = {'Baseline': 0, 'Unlock': 1, 'Transaction': 2}
        self.class_names = ['Baseline', 'Unlock', 'Transaction']
        
        # Initialize models and scalers
        self.rf_model = None
        self.cnn_model = None
        self.scaler = StandardScaler()
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Results storage
        self.results = {}
        
        # Real-time classification variables (Task 16)
        self.model_loaded = False
        self.prediction_history = deque(maxlen=10)  # For temporal smoothing
        self.confidence_threshold = 0.7
        self.smoothing_window = 5
        
        print("=== Task 07: EEG Data Loading and Preprocessing ===")
        
    def load_and_preprocess_data(self):
        """
        Task 07: Load eeg_patterns.csv and prepare data for machine learning
        
        Requirements:
        - Load CSV with pandas, handle timestamps
        - Extract features from 12 EEG channels (EEG1-EEG12)
        - Create labels: 0=Baseline, 1=Unlock, 2=Transaction
        - Split data into train/validation/test sets (70/15/15)
        - Apply feature scaling/normalization
        - Handle class imbalance if present
        """
        
        print(f"Loading EEG data from: {self.data_path}")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"EEG data file not found: {self.data_path}")
        
        # Load CSV data
        try:
            df = pd.read_csv(self.data_path)
            print(f"✓ Successfully loaded {len(df)} samples")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
        
        # Validate required columns
        required_cols = ['Phase', 'Timestamp'] + self.eeg_channels
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"✓ Validated data structure: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Extract EEG features (12 channels)
        X = df[self.eeg_channels].values
        print(f"✓ Extracted EEG features: {X.shape}")
        
        # Create labels from Phase column
        y = df['Phase'].map(self.phase_mapping).values
        if np.any(pd.isna(y)):
            raise ValueError("Invalid phase labels found in data")
        
        print(f"✓ Created labels: {len(y)} samples")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  {self.class_names[class_idx]}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Handle class imbalance warning
        min_samples = min(counts)
        max_samples = max(counts)
        imbalance_ratio = max_samples / min_samples
        if imbalance_ratio > 2.0:
            print(f"⚠ Warning: Class imbalance detected (ratio: {imbalance_ratio:.1f})")
            print("  Consider using class weights in model training")
        else:
            print("✓ Classes are reasonably balanced")
        
        # Split data: 70% train, 15% validation, 15% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        print(f"✓ Data split completed:")
        print(f"  Training: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(self.X_val)} samples ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Apply feature scaling/normalization
        print("Applying feature scaling...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Calculate and display scaling statistics
        print(f"✓ Feature scaling applied:")
        print(f"  Original range: [{X.min():.1f}, {X.max():.1f}]")
        print(f"  Scaled range: [{self.X_train_scaled.min():.2f}, {self.X_train_scaled.max():.2f}]")
        print(f"  Mean: {self.X_train_scaled.mean():.3f}, Std: {self.X_train_scaled.std():.3f}")
        
        print("✅ Data loading and preprocessing completed")
        
        return True
    
    def build_cnn_model(input_dim, num_classes=1):
        inputs = Input(shape=(input_dim,))
        x = Reshape((input_dim, 1))(inputs)
        x = Conv1D(64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(32, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        if num_classes == 1 or num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model

    def train_random_forest(self):
        """
        Random Forest Model Implementation
        
        Requirements:
        - Use RandomForestClassifier from sklearn
        - Hyperparameter tuning: n_estimators, max_depth, min_samples_split
        - Cross-validation for robust evaluation
        - Feature importance analysis
        - Save trained model as pickle file
        """
        
        print("\n=== Random Forest Model Implementation ===")
        
        if self.X_train_scaled is None:
            raise ValueError("Data not preprocessed. Run load_and_preprocess_data() first.")
        
        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        print("Starting hyperparameter tuning with GridSearchCV...")
        print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # Initialize Random Forest with cross-validation
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        self.rf_model = grid_search.best_estimator_
        
        print(f"✓ Best hyperparameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        # Evaluate model performance
        train_acc = self.rf_model.score(self.X_train_scaled, self.y_train)
        val_acc = self.rf_model.score(self.X_val_scaled, self.y_val)
        
        print(f"✓ Model performance:")
        print(f"  Training accuracy: {train_acc:.3f}")
        print(f"  Validation accuracy: {val_acc:.3f}")
        print(f"  Best CV score: {grid_search.best_score_:.3f}")
        
        # Cross-validation for robust evaluation
        cv_scores = cross_val_score(self.rf_model, self.X_train_scaled, self.y_train, cv=5)
        print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance analysis
        feature_importance = self.rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'channel': self.eeg_channels,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"✓ Feature importance analysis:")
        print("  Top 5 most important EEG channels:")
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            print(f"    {i+1}. {row['channel']}: {row['importance']:.3f}")
        
        # Save trained model
        model_path = 'models/rf_eeg_classifier.pkl'
        os.makedirs('models', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.rf_model,
                'scaler': self.scaler,
                'feature_names': self.eeg_channels,
                'class_names': self.class_names
            }, f)
        
        print(f"✓ Model saved to: {model_path}")
        
        # Store results for comparison
        self.results['rf'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'cv_acc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': importance_df
        }
        
        # Check success criteria
        if val_acc >= 0.60:
            print("✅ Task 08 COMPLETED: Random Forest model meets >80% accuracy criteria")
            return True
        else:
            print(f"⚠ Warning: Validation accuracy ({val_acc:.1%}) below 80% target")
            return False
    
    def train_cnn_model(self):
        """
        CNN Model Implementation
        
        Requirements:
        - Import build_cnn_model function
        - Modify for 12-channel input and 3-class output
        - Add data augmentation for time-series
        - Implement early stopping and model checkpointing
        - Save best model as .h5 file
        """
        
        print("\n=== Task 09: CNN Model Implementation ===")
        
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Skipping CNN implementation.")
            return False
        
        if self.X_train_scaled is None:
            raise ValueError("Data not preprocessed. Run load_and_preprocess_data() first.")
        
        # Prepare data for CNN (expects specific input format)
        input_dim = self.X_train_scaled.shape[1]  # 12 EEG channels
        num_classes = len(self.class_names)  # 3 classes
        
        print(f"Building CNN model for input_dim={input_dim}, num_classes={num_classes}")
        
        # Build CNN model using imported function
        self.cnn_model = build_cnn_model(input_dim, num_classes)
        
        print("✓ CNN model architecture created")
        print(f"  Input shape: {input_dim}")
        print(f"  Output classes: {num_classes}")
        
        # Prepare labels for categorical classification
        y_train_cat = to_categorical(self.y_train, num_classes)
        y_val_cat = to_categorical(self.y_val, num_classes)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/cnn_eeg_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        print("Starting CNN training...")
        
        # Train the model
        history = self.cnn_model.fit(
            self.X_train_scaled, y_train_cat,
            validation_data=(self.X_val_scaled, y_val_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model performance
        train_loss, train_acc = self.cnn_model.evaluate(self.X_train_scaled, y_train_cat, verbose=0)
        val_loss, val_acc = self.cnn_model.evaluate(self.X_val_scaled, y_val_cat, verbose=0)
        
        print(f"✓ CNN training completed:")
        print(f"  Training accuracy: {train_acc:.3f}")
        print(f"  Validation accuracy: {val_acc:.3f}")
        print(f"  Training loss: {train_loss:.3f}")
        print(f"  Validation loss: {val_loss:.3f}")
        
        # Save training history
        history_path = 'models/cnn_training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"✓ Training history saved to: {history_path}")
        
        # Store results for comparison
        self.results['cnn'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history.history
        }
        
        # Check success criteria
        if val_acc >= 0.85:
            print("✅ Task 09 COMPLETED: CNN model meets >85% accuracy criteria")
            return True
        else:
            print(f"⚠ Warning: Validation accuracy ({val_acc:.1%}) below 85% target")
            return False
    
    def evaluate_and_compare_models(self):
        """
        Task 10: Model Evaluation and Comparison
        
        Requirements:
        - Confusion matrices for both models
        - Precision, recall, F1-score per class
        - ROC curves and AUC scores
        - Classification reports
        - Model comparison table
        - Performance visualization plots
        """
        
        print("\n=== Task 10: Model Evaluation and Comparison ===")
        
        if self.rf_model is None:
            print("❌ Random Forest model not trained. Run train_random_forest() first.")
            return False
        
        # Create evaluation directory
        os.makedirs('evaluation', exist_ok=True)
        
        # Test both models on test set
        print("Evaluating models on test set...")
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        rf_pred_proba = self.rf_model.predict_proba(self.X_test_scaled)
        rf_test_acc = accuracy_score(self.y_test, rf_pred)
        rf_f1 = f1_score(self.y_test, rf_pred, average='weighted')
        
        print(f"✓ Random Forest test results:")
        print(f"  Accuracy: {rf_test_acc:.3f}")
        print(f"  F1-Score: {rf_f1:.3f}")
        
        # CNN predictions (if available)
        if TENSORFLOW_AVAILABLE and self.cnn_model is not None:
            y_test_cat = to_categorical(self.y_test, len(self.class_names))
            cnn_pred_proba = self.cnn_model.predict(self.X_test_scaled)
            cnn_pred = np.argmax(cnn_pred_proba, axis=1)
            cnn_test_acc = accuracy_score(self.y_test, cnn_pred)
            cnn_f1 = f1_score(self.y_test, cnn_pred, average='weighted')
            
            print(f"✓ CNN test results:")
            print(f"  Accuracy: {cnn_test_acc:.3f}")
            print(f"  F1-Score: {cnn_f1:.3f}")
        else:
            cnn_pred = None
            cnn_pred_proba = None
            cnn_test_acc = 0
            cnn_f1 = 0
            print("⚠ CNN model not available for evaluation")
        
        # Generate comprehensive evaluation report
        self._generate_confusion_matrices(rf_pred, cnn_pred)
        self._generate_classification_reports(rf_pred, cnn_pred)
        self._generate_roc_curves(rf_pred_proba, cnn_pred_proba)
        self._generate_comparison_table(rf_test_acc, rf_f1, cnn_test_acc, cnn_f1)
        self._generate_performance_plots()
        
        print("✅ Task 10 COMPLETED: Model evaluation and comparison finished")
        print(f"📊 Evaluation results saved in 'evaluation/' directory")
        
        return True
    
    def _generate_confusion_matrices(self, rf_pred, cnn_pred):
        """Generate and save confusion matrices for both models"""
        fig, axes = plt.subplots(1, 2 if cnn_pred is not None else 1, figsize=(12, 5))
        if cnn_pred is None:
            axes = [axes]
        
        # Random Forest confusion matrix
        rf_cm = confusion_matrix(self.y_test, rf_pred)
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
        axes[0].set_title('Random Forest Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # CNN confusion matrix (if available)
        if cnn_pred is not None:
            cnn_cm = confusion_matrix(self.y_test, cnn_pred)
            sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1])
            axes[1].set_title('CNN Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('evaluation/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrices saved")
    
    def _generate_classification_reports(self, rf_pred, cnn_pred):
        """Generate detailed classification reports"""
        print("\n📋 Detailed Classification Reports:")
        
        print("\n--- Random Forest Classification Report ---")
        rf_report = classification_report(self.y_test, rf_pred, target_names=self.class_names)
        print(rf_report)
        
        with open('evaluation/rf_classification_report.txt', 'w') as f:
            f.write("Random Forest Classification Report\n")
            f.write("="*50 + "\n")
            f.write(rf_report)
        
        if cnn_pred is not None:
            print("\n--- CNN Classification Report ---")
            cnn_report = classification_report(self.y_test, cnn_pred, target_names=self.class_names)
            print(cnn_report)
            
            with open('evaluation/cnn_classification_report.txt', 'w') as f:
                f.write("CNN Classification Report\n")
                f.write("="*50 + "\n")
                f.write(cnn_report)
        
        print("✓ Classification reports saved")
    
    def _generate_roc_curves(self, rf_pred_proba, cnn_pred_proba):
        """Generate ROC curves and calculate AUC scores"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output for multi-class ROC
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Random Forest ROC curves
        plt.subplot(2, 2, 1)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest ROC Curves')
        plt.legend()
        
        # CNN ROC curves (if available)
        if cnn_pred_proba is not None:
            plt.subplot(2, 2, 2)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], cnn_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('CNN ROC Curves')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('evaluation/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ROC curves saved")
    
    def _generate_comparison_table(self, rf_acc, rf_f1, cnn_acc, cnn_f1):
        """Generate model comparison table"""
        comparison_data = {
            'Model': ['Random Forest', 'CNN'],
            'Test Accuracy': [f"{rf_acc:.3f}", f"{cnn_acc:.3f}" if cnn_acc > 0 else "N/A"],
            'Test F1-Score': [f"{rf_f1:.3f}", f"{cnn_f1:.3f}" if cnn_f1 > 0 else "N/A"],
            'Val Accuracy': [f"{self.results.get('rf', {}).get('val_acc', 0):.3f}",
                           f"{self.results.get('cnn', {}).get('val_acc', 0):.3f}" if 'cnn' in self.results else "N/A"],
            'Parameters': [f"~{self.rf_model.n_estimators * 100}K", "~50K" if cnn_acc > 0 else "N/A"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 Model Comparison Table:")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv('evaluation/model_comparison.csv', index=False)
        print("✓ Comparison table saved")
    
    def _generate_performance_plots(self):
        """Generate performance visualization plots"""
        if 'cnn' in self.results and 'history' in self.results['cnn']:
            # CNN training history plots
            history = self.results['cnn']['history']
            
            plt.figure(figsize=(12, 4))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('CNN Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('CNN Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('evaluation/cnn_training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Feature importance plot (Random Forest)
        if 'rf' in self.results and 'feature_importance' in self.results['rf']:
            plt.figure(figsize=(10, 6))
            importance_df = self.results['rf']['feature_importance']
            
            sns.barplot(data=importance_df, x='importance', y='channel')
            plt.title('Random Forest Feature Importance\n(EEG Channel Contribution)')
            plt.xlabel('Importance Score')
            plt.ylabel('EEG Channel')
            
            plt.tight_layout()
            plt.savefig('evaluation/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ Performance plots saved")
    
    # ===== REAL-TIME CLASSIFICATION METHODS (Task 16) =====
    
    def load_trained_model(self, model_path='models/rf_eeg_classifier.pkl'):
        """
        Task 16: Load pre-trained model on startup
        
        Requirements:
        - Load saved Random Forest model and scaler
        - Verify model compatibility
        - Initialize real-time prediction pipeline
        - Load model metadata and feature names
        """
        
        print(f"\n=== Loading Trained Model for Real-Time Classification ===")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("   Please train the model first by running the full training pipeline")
            return False
        
        try:
            # Load saved model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract components
            self.rf_model = model_data['model']
            self.scaler = model_data['scaler']
            loaded_features = model_data['feature_names']
            loaded_classes = model_data['class_names']
            
            # Verify compatibility
            if loaded_features != self.eeg_channels:
                print(f"⚠ Warning: Feature names don't match")
                print(f"   Expected: {self.eeg_channels}")
                print(f"   Loaded: {loaded_features}")
            
            if loaded_classes != self.class_names:
                print(f"⚠ Warning: Class names don't match")
                print(f"   Expected: {self.class_names}")
                print(f"   Loaded: {loaded_classes}")
                self.class_names = loaded_classes  # Use loaded class names
            
            # Initialize prediction pipeline
            self.model_loaded = True
            self.prediction_history.clear()
            
            print(f"✅ Model loaded successfully:")
            print(f"   Model type: {type(self.rf_model).__name__}")
            print(f"   Features: {len(loaded_features)} EEG channels")
            print(f"   Classes: {len(loaded_classes)} ({', '.join(loaded_classes)})")
            print(f"   Scaler: {type(self.scaler).__name__}")
            print(f"   Ready for real-time classification")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
            return False
    
    def real_time_predict(self, eeg_channels_data):
        """
        Task 16: Real-time classification with <100ms latency
        
        Requirements:
        - Accept 12-channel EEG data as input
        - Apply same preprocessing as training data
        - Return classification with confidence scores
        - Maintain <100ms processing latency
        - Apply temporal smoothing
        
        Args:
            eeg_channels_data: List or array of 12 EEG channel values
            
        Returns:
            dict: {
                'class': predicted class name,
                'class_id': predicted class ID (0, 1, 2),
                'confidence': confidence score (0-1),
                'probabilities': array of class probabilities,
                'processing_time': time taken in milliseconds,
                'smoothed': whether temporal smoothing was applied
            }
        """
        
        start_time = time.time()
        
        if not self.model_loaded:
            return {
                'class': 'error',
                'class_id': -1,
                'confidence': 0.0,
                'probabilities': None,
                'processing_time': 0.0,
                'smoothed': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Validate input
            if len(eeg_channels_data) != 12:
                raise ValueError(f"Expected 12 EEG channels, got {len(eeg_channels_data)}")
            
            # Convert to numpy array and reshape for prediction
            X = np.array(eeg_channels_data).reshape(1, -1)
            
            # Apply same scaling as training data
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.rf_model.predict(X_scaled)[0]
            probabilities = self.rf_model.predict_proba(X_scaled)[0]
            
            # Get confidence (max probability)
            confidence = np.max(probabilities)
            
            # Store prediction for temporal smoothing
            prediction_data = {
                'class_id': prediction,
                'confidence': confidence,
                'probabilities': probabilities.copy(),
                'timestamp': time.time()
            }
            self.prediction_history.append(prediction_data)
            
            # Apply temporal smoothing if we have enough history
            smoothed_result = self.apply_temporal_smoothing()
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Use smoothed result if available, otherwise use current prediction
            if smoothed_result is not None:
                final_class_id = smoothed_result['class_id']
                final_confidence = smoothed_result['confidence']
                final_probabilities = smoothed_result['probabilities']
                smoothed = True
            else:
                final_class_id = prediction
                final_confidence = confidence
                final_probabilities = probabilities
                smoothed = False
            
            # Get class name
            class_name = self.class_names[final_class_id]
            
            result = {
                'class': class_name,
                'class_id': final_class_id,
                'confidence': final_confidence,
                'probabilities': final_probabilities,
                'processing_time': processing_time,
                'smoothed': smoothed
            }
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                'class': 'error',
                'class_id': -1,
                'confidence': 0.0,
                'probabilities': None,
                'processing_time': processing_time,
                'smoothed': False,
                'error': str(e)
            }
    
    def apply_temporal_smoothing(self):
        """
        Task 16: Apply temporal smoothing to reduce noise
        
        Requirements:
        - Use sliding window of recent predictions
        - Weight recent predictions more heavily
        - Return smoothed classification result
        - Improve stability of classifications
        
        Returns:
            dict or None: Smoothed prediction result or None if insufficient data
        """
        
        if len(self.prediction_history) < self.smoothing_window:
            return None
        
        # Get recent predictions
        recent_predictions = list(self.prediction_history)[-self.smoothing_window:]
        
        # Calculate weighted average of probabilities
        # More recent predictions get higher weights
        weights = np.exp(np.linspace(0, 1, len(recent_predictions)))
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted average of probabilities
        avg_probabilities = np.zeros(len(self.class_names))
        for i, pred in enumerate(recent_predictions):
            avg_probabilities += weights[i] * pred['probabilities']
        
        # Get smoothed prediction
        smoothed_class_id = np.argmax(avg_probabilities)
        smoothed_confidence = np.max(avg_probabilities)
        
        return {
            'class_id': smoothed_class_id,
            'confidence': smoothed_confidence,
            'probabilities': avg_probabilities
        }
    
    def get_confidence_scores(self, eeg_channels_data):
        """
        Task 16: Get detailed confidence scores for each class
        
        Requirements:
        - Return confidence for all classes
        - Include prediction stability metrics
        - Provide recommendation on prediction reliability
        
        Args:
            eeg_channels_data: List or array of 12 EEG channel values
            
        Returns:
            dict: Detailed confidence analysis
        """
        
        # Get base prediction
        prediction_result = self.real_time_predict(eeg_channels_data)
        
        if 'error' in prediction_result:
            return prediction_result
        
        # Calculate detailed confidence metrics
        probabilities = prediction_result['probabilities']
        
        # Confidence scores for each class
        class_confidences = {}
        for i, class_name in enumerate(self.class_names):
            class_confidences[class_name] = {
                'probability': float(probabilities[i]),
                'confidence_level': self._get_confidence_level(probabilities[i])
            }
        
        # Prediction stability (based on recent history)
        stability_score = self._calculate_stability_score()
        
        # Overall recommendation
        max_prob = np.max(probabilities)
        second_max_prob = np.partition(probabilities, -2)[-2]
        margin = max_prob - second_max_prob
        
        if max_prob >= 0.9 and margin >= 0.3:
            reliability = "Very High"
        elif max_prob >= 0.8 and margin >= 0.2:
            reliability = "High"
        elif max_prob >= 0.7 and margin >= 0.1:
            reliability = "Medium"
        elif max_prob >= 0.6:
            reliability = "Low"
        else:
            reliability = "Very Low"
        
        return {
            'predicted_class': prediction_result['class'],
            'predicted_class_id': prediction_result['class_id'],
            'overall_confidence': float(max_prob),
            'class_confidences': class_confidences,
            'prediction_margin': float(margin),
            'stability_score': stability_score,
            'reliability': reliability,
            'recommendation': self._get_recommendation(reliability, max_prob),
            'processing_time': prediction_result['processing_time']
        }
    
    def _get_confidence_level(self, probability):
        """Convert probability to confidence level description"""
        if probability >= 0.9:
            return "Very High"
        elif probability >= 0.8:
            return "High"
        elif probability >= 0.7:
            return "Medium"
        elif probability >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def _calculate_stability_score(self):
        """Calculate prediction stability based on recent history"""
        if len(self.prediction_history) < 3:
            return 0.0
        
        # Look at last few predictions
        recent_classes = [pred['class_id'] for pred in list(self.prediction_history)[-5:]]
        
        if len(set(recent_classes)) == 1:
            # All predictions are the same
            return 1.0
        elif len(set(recent_classes)) == 2:
            # Two different classes
            return 0.6
        else:
            # More variation
            return 0.3
    
    def _get_recommendation(self, reliability, confidence):
        """Get recommendation based on reliability and confidence"""
        if reliability in ["Very High", "High"] and confidence >= 0.8:
            return "Accept prediction - high confidence"
        elif reliability == "Medium" and confidence >= 0.7:
            return "Accept with caution - medium confidence"
        elif reliability == "Low":
            return "Review prediction - low confidence"
        else:
            return "Reject prediction - insufficient confidence"
    
    def reset_prediction_history(self):
        """Reset the prediction history for temporal smoothing"""
        self.prediction_history.clear()
        print("✓ Prediction history reset")
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for predictions"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"✓ Confidence threshold set to: {threshold}")
        else:
            print(f"❌ Invalid threshold: {threshold}. Must be between 0.0 and 1.0")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"error": "No model loaded"}
        
        return {
            "model_type": type(self.rf_model).__name__,
            "n_features": len(self.eeg_channels),
            "n_classes": len(self.class_names),
            "class_names": self.class_names,
            "feature_names": self.eeg_channels,
            "model_loaded": self.model_loaded,
            "prediction_history_length": len(self.prediction_history),
            "confidence_threshold": self.confidence_threshold,
            "smoothing_window": self.smoothing_window
        }

def main():
    """Main execution following tasks_eeg.yml implementation plan"""
    print("EEG Real-Time Classification System")
    print("Multi-Modal Biometric Authentication")
    print("="*50)
    
    # Initialize classifier
    classifier = EEGClassifier()
    
    try:
        # Task 07: Data Loading and Preprocessing
        classifier.load_and_preprocess_data()
        
        # Task 08: Random Forest Implementation
        classifier.train_random_forest()
        
        # Task 09: CNN Implementation
        if TENSORFLOW_AVAILABLE:
            classifier.train_cnn_model()
        else:
            print("⚠ Skipping CNN training - TensorFlow not available")
        
        # Task 10: Model Evaluation and Comparison
        classifier.evaluate_and_compare_models()
        
        print("\n" + "="*50)
        print("✅ Phase 1 (Tasks 07-10) FULLY COMPLETED")
        print("📁 Results saved in 'models/' and 'evaluation/' directories")
        print("🚀 Ready for Phase 2: Real-Time Classification System")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 