import sys
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import os

def main():
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument("--model_path", required=True, help="Path to the saved model file")
    parser.add_argument("--input_values", required=True, help="JSON object with feature names and values")
    
    args = parser.parse_args()
    
    try:
        # Load model, scaler, and preprocessing info
        model = joblib.load(args.model_path)
        model_dir = os.path.dirname(args.model_path)
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        preprocessing_path = os.path.join(model_dir, "preprocessing.pkl")
        
        # Load scaler if exists
        has_scaler = os.path.exists(scaler_path)
        if has_scaler:
            scaler = joblib.load(scaler_path)
        else:
            print("WARNING: Scaler not found, predicting without scaling", file=sys.stderr)
        
        # Load preprocessing info
        if not os.path.exists(preprocessing_path):
            print("ERROR: Preprocessing info not found. Please retrain the model.", file=sys.stderr)
            sys.exit(1)
        
        preprocessing_info = joblib.load(preprocessing_path)
        original_train_columns = preprocessing_info.get('original_train_columns', [])
        
        # Check if this is an old model without final_feature_names
        if 'final_feature_names' not in preprocessing_info:
            print("WARNING: Old model format detected. Please retrain for best results.", file=sys.stderr)
            print("ERROR: This model was trained with an older version. Please retrain the model to use predictions.", file=sys.stderr)
            print("RETRAIN_REQUIRED", file=sys.stderr)
            sys.exit(1)
        
        final_feature_names = preprocessing_info['final_feature_names']
        encoding_type = preprocessing_info.get('encoding_type', 'one-hot')
        categorical_cols = preprocessing_info.get('categorical_cols', [])
        label_encoders = preprocessing_info.get('label_encoders', {})
        target_means = preprocessing_info.get('target_means', {})
        encoding_added_cols = preprocessing_info.get('encoding_added_cols', {})
        is_binary = preprocessing_info.get('is_binary_classification', False)
        target_name = preprocessing_info.get('target_name', 'target')
        
        # Determine numeric columns
        numeric_cols = [col for col in original_train_columns if col not in categorical_cols]
        
        # Check if final_feature_names seems incorrect for one-hot encoding
        if (encoding_type.lower() in ['onehot', 'one-hot']) and len(categorical_cols) > 0:
            # If using one-hot encoding with categorical columns, final_feature_names should be expanded
            if len(final_feature_names) == len(original_train_columns):
                print("=" * 80, file=sys.stderr)
                print("❌ ERROR: OLD MODEL FORMAT DETECTED", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print("This model was trained with an older version.", file=sys.stderr)
                print("The preprocessing info doesn't contain expanded one-hot column names.", file=sys.stderr)
                print("", file=sys.stderr)
                print("🔄 SOLUTION: Please retrain the model to use predictions.", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print("RETRAIN_REQUIRED", file=sys.stderr)
                sys.exit(1)
        
        # Parse input values (should be a dict: {feature_name: value})
        input_data = json.loads(args.input_values)
        
        print(f"DEBUG: Input data: {input_data}", file=sys.stderr)
        print(f"DEBUG: Encoding type: {encoding_type}", file=sys.stderr)
        print(f"DEBUG: Categorical columns: {categorical_cols}", file=sys.stderr)
        print(f"DEBUG: Expected final features: {len(final_feature_names)} features", file=sys.stderr)
        
        # Create DataFrame with original columns
        df = pd.DataFrame([input_data], columns=original_train_columns)
        
        # Apply same encoding as during training
        encoding_type_normalized = encoding_type.lower().replace('-', '')  # 'one-hot' -> 'onehot'
        if encoding_type_normalized in ['onehot']:
            print(f"DEBUG: Applying One-Hot encoding", file=sys.stderr)
            print(f"DEBUG: DataFrame before encoding:\n{df.to_dict()}", file=sys.stderr)
            
            # DON'T use pd.get_dummies() - it creates unpredictable column names
            # Instead, manually create dummy columns based on final_feature_names
            
            # First, separate numeric columns and convert them to proper numeric type
            numeric_cols = [col for col in original_train_columns if col not in categorical_cols]
            if numeric_cols:
                df_numeric = df[numeric_cols].copy()
                # Convert numeric columns from string to float
                for col in numeric_cols:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                print(f"DEBUG: Converted {len(numeric_cols)} numeric columns to float", file=sys.stderr)
            else:
                df_numeric = pd.DataFrame()
            
            # Create all one-hot encoded columns from final_feature_names
            # Initialize all one-hot columns to 0
            onehot_data = {}
            numeric_feature_names = [f for f in final_feature_names if f in numeric_cols]
            onehot_feature_names = [f for f in final_feature_names if f not in numeric_cols]
            
            # Initialize all one-hot columns to 0
            for onehot_col in onehot_feature_names:
                onehot_data[onehot_col] = 0
            
            # Set the appropriate one-hot columns to 1 based on input values
            for cat_col in categorical_cols:
                input_value = str(df[cat_col].values[0])
                expected_col_name = f"{cat_col}_{input_value}"
                
                # Set to 1 if this column exists in final_feature_names
                if expected_col_name in onehot_data:
                    onehot_data[expected_col_name] = 1
                    print(f"DEBUG: Set {expected_col_name} = 1", file=sys.stderr)
                else:
                    print(f"DEBUG: {expected_col_name} not in final features (likely dropped with drop_first=True)", file=sys.stderr)
            
            # Combine numeric and one-hot encoded columns
            onehot_df = pd.DataFrame([onehot_data])
            df = pd.concat([df_numeric.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)
            
            print(f"DEBUG: DataFrame after manual one-hot - Columns: {df.columns.tolist()}", file=sys.stderr)
            print(f"DEBUG: Non-zero values: {(df != 0).sum().sum()}", file=sys.stderr)
            
        elif encoding_type_normalized == "label":
            print(f"DEBUG: Applying Label encoding", file=sys.stderr)
            
            # Apply label encoding using saved encoders
            for col in categorical_cols:
                if col in df.columns and col in label_encoders:
                    le = label_encoders[col]
                    input_value = str(df[col].values[0])
                    print(f"DEBUG: Encoding '{col}' = '{input_value}'", file=sys.stderr)
                    
                    try:
                        encoded_value = le.transform([input_value])[0]
                        df[col] = encoded_value
                        print(f"DEBUG: Encoded to {encoded_value}", file=sys.stderr)
                    except ValueError:
                        print(f"WARNING: Unknown category '{input_value}' in column '{col}'. Using fallback value 0.", file=sys.stderr)
                        df[col] = 0
            
            # Convert numeric columns to proper numeric type
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"DEBUG: Label encoding completed. Final shape: {df.shape}", file=sys.stderr)
                        
        elif encoding_type_normalized == "target":
            print(f"DEBUG: Applying Target encoding", file=sys.stderr)
            
            if target_means:
                # Apply target encoding using saved mean mappings
                for col in categorical_cols:
                    if col in df.columns and col in target_means:
                        input_value = str(df[col].values[0])
                        mean_mappings = target_means[col]
                        
                        print(f"DEBUG: Target encoding '{col}' = '{input_value}'", file=sys.stderr)
                        
                        if input_value in mean_mappings:
                            encoded_value = mean_mappings[input_value]
                            df[col] = encoded_value
                            print(f"DEBUG: Mapped to mean value: {encoded_value}", file=sys.stderr)
                        else:
                            # Unknown category - use overall mean or 0
                            fallback_value = sum(mean_mappings.values()) / len(mean_mappings) if mean_mappings else 0
                            df[col] = fallback_value
                            print(f"WARNING: Unknown category '{input_value}' - using mean fallback: {fallback_value}", file=sys.stderr)
            else:
                print("WARNING: Target encoding mappings not found. Using fallback value 0.", file=sys.stderr)
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = 0
            
            # Convert numeric columns to proper numeric type
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"DEBUG: Target encoding completed. Final shape: {df.shape}", file=sys.stderr)
                    
        elif encoding_type_normalized == "none":
            print(f"DEBUG: No encoding applied - using only numeric features", file=sys.stderr)
            
            # Convert all numeric columns
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only numeric columns
            df = df[numeric_cols]
            print(f"DEBUG: Using {len(numeric_cols)} numeric columns: {numeric_cols}", file=sys.stderr)
        
        # Ensure all expected features are present
        missing_features = []
        for feature in final_feature_names:
            if feature not in df.columns:
                # Add missing feature with 0 (happens with one-hot when category wasn't present)
                df[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            print(f"DEBUG: Added {len(missing_features)} missing features with value 0", file=sys.stderr)
            print(f"DEBUG: First 10 missing: {missing_features[:10]}", file=sys.stderr)
        
        # Select and reorder columns to match training
        df = df[final_feature_names]
        
        print(f"DEBUG: Final DataFrame shape: {df.shape}", file=sys.stderr)
        print(f"DEBUG: Final DataFrame (first 10 values): {df.values[0][:10]}", file=sys.stderr)
        print(f"DEBUG: Non-zero feature count: {np.count_nonzero(df.values)}", file=sys.stderr)
        
        # Convert to numpy array
        X_input = df.values
        
        # Scale the input if scaler is available
        if has_scaler:
            print(f"DEBUG: Before scaling - min: {X_input.min()}, max: {X_input.max()}", file=sys.stderr)
            X_input = scaler.transform(X_input)
            print(f"DEBUG: After scaling - min: {X_input.min()}, max: {X_input.max()}", file=sys.stderr)
        
        # Make prediction
        prediction = model.predict(X_input)
        prediction_value = prediction[0]
        
        print(f"DEBUG: Raw prediction value: {prediction_value}", file=sys.stderr)
        print(f"DEBUG: Target '{target_name}' is binary classification: {is_binary}", file=sys.stderr)
        
        # Check if we need to decode the prediction using label_encoder
        label_encoder = preprocessing_info.get('label_encoder', None)
        if label_encoder is not None:
            # Decode the prediction back to original label
            try:
                decoded_prediction = label_encoder.inverse_transform([int(prediction_value)])[0]
                print(f"DEBUG: Decoded prediction: {prediction_value} → {decoded_prediction}", file=sys.stderr)
                final_value = decoded_prediction
            except Exception as e:
                print(f"WARNING: Could not decode prediction: {str(e)}", file=sys.stderr)
                final_value = float(prediction_value)
        else:
            # Apply clipping based on training target type
            if is_binary:
                # Binary classification - clip to [0, 1] probability range
                final_value = float(np.clip(prediction_value, 0.0, 1.0))
                print(f"DEBUG: Applied binary classification clipping: {prediction_value} → {final_value}", file=sys.stderr)
            else:
                # Regression - keep raw value
                final_value = float(prediction_value)
                print(f"DEBUG: Regression - using raw prediction: {final_value}", file=sys.stderr)
        
        print(f"DEBUG: Final prediction value: {final_value}", file=sys.stderr)
        
        # Output the prediction in a parseable format
        # Include metadata for frontend to properly display result
        print(f"PREDICTION:{final_value}")
        print(f"IS_BINARY:{is_binary}", file=sys.stderr)
        sys.stdout.flush()
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {str(e)}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing feature in input data - {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
