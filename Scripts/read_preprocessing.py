import sys
import joblib
import json

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Preprocessing file path required"}))
        sys.exit(1)
    
    preprocessing_path = sys.argv[1]
    
    try:
        preprocessing_info = joblib.load(preprocessing_path)
        
        # Extract relevant info for frontend
        original_columns = preprocessing_info.get('original_train_columns', [])
        categorical_cols = preprocessing_info.get('categorical_cols', [])
        numeric_cols = [col for col in original_columns if col not in categorical_cols]
        
        result = {
            "original_train_columns": original_columns,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "categorical_values": preprocessing_info.get('categorical_values', {}),
            "encoding_type": preprocessing_info.get('encoding_type', 'none')
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
