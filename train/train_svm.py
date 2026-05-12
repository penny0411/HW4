import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_images_from_folder(folder_path):
    """Read images from local folder and convert to SVM-ready array"""
    images = []
    labels = []
    # Define label mapping (must match demo side: 0=Rock, 1=Paper, 2=Scissors)
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for category, label_idx in label_map.items():
        # Path e.g. dataset/train/rock
        category_path = os.path.join(folder_path, category)
        
        # Handle case where there might be an extra folder layer
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Cannot find {category} folder -> {category_path}")
            continue
            
        print(f"Loading {category} images...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 1. Grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 2. Resize to 64x64
                    resized = cv2.resize(gray, (64, 64))
                    # 3. Flatten to 1D array
                    images.append(resized.flatten())
                    labels.append(label_idx)
                    
    # Normalize pixel values
    return np.array(images) / 255.0, np.array(labels)

def main():
    # Get project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== Step 1: Loading images ===")
    X_train, y_train = load_images_from_folder(train_dir)
    print("---")
    X_test, y_test = load_images_from_folder(test_dir)

    print(f"\nRead complete! Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    if len(X_train) == 0:
        print("Error: No images found. Check dataset folder structure.")
        return

    print("\n=== Step 2: Training SVM model (may take 1-2 mins) ===")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    print("\n=== Step 3: Evaluating model ===")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    # Ensure demo folder exists
    os.makedirs(demo_dir, exist_ok=True)
    
    # Save model to demo folder
    model_path = os.path.join(demo_dir, 'rps_svm_model.pkl')
    joblib.dump(clf, model_path)
    print(f"Model saved successfully: {model_path}")

if __name__ == "__main__":
    main()